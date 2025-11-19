import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import pandas as pd

class E5Embedder:
    def __init__(self, model_name: str = 'intfloat/e5-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 768
        self.cache = {}
    
    def embed(self, text: str, transaction_mode: str = '', 
              recipient_name: str = None, upi_id: str = None, 
              note: str = None, amount: float = None) -> np.ndarray:
        """
        Generate E5 embedding for transaction with enhanced UPI data.
        
        Args:
            text: Normalized description/merchant text
            transaction_mode: Payment mode (UPI, NEFT, etc.)
            recipient_name: UPI recipient name (e.g., "JULFIKAR", "NETFLIX")
            upi_id: UPI ID (e.g., "paytmqr1jc", "netflixupi")
            note: Transaction note (e.g., "baker", "Monthly", "UPI")
            amount: Transaction amount (for context)
            
        Returns: 768-dim L2-normalized vector
        """
        if not text:
            text = ''
        
        # Build rich context string
        context_parts = [text]
        
        # Add recipient name (often most informative)
        if recipient_name and not pd.isna(recipient_name):
            recipient_clean = str(recipient_name).strip()
            if recipient_clean and recipient_clean.lower() not in text.lower():
                context_parts.append(recipient_clean)
        
        # Add note (provides category hints)
        if note and not pd.isna(note):
            note_clean = str(note).strip()
            if note_clean and len(note_clean) > 2:
                context_parts.append(note_clean)
        
        # Add UPI ID (can contain merchant info)
        if upi_id and not pd.isna(upi_id):
            upi_clean = str(upi_id).strip()
            # Extract meaningful parts (e.g., "paytmqr" from "paytmqr1jc")
            upi_base = ''.join([c for c in upi_clean if c.isalpha()])
            if upi_base and len(upi_base) > 3 and upi_base not in text.lower():
                context_parts.append(upi_base)
        
        # Add transaction mode
        if transaction_mode:
            context_parts.append(transaction_mode)
        
        # Add amount range (for context: small/medium/large transaction)
        if amount and amount > 0:
            if amount < 100:
                context_parts.append("small payment")
            elif amount < 1000:
                context_parts.append("medium transaction")
            elif amount < 10000:
                context_parts.append("large payment")
            else:
                context_parts.append("very large transaction")
        
        # Combine all context
        rich_text = ' '.join(context_parts)
        
        # Check cache
        cache_key = f"{rich_text}_{transaction_mode}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # E5 requires "query: " prefix
        query = f"query: {rich_text}".strip()
        
        # Generate embedding
        embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Cache result
        self.cache[cache_key] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str], modes: List[str] = None,
                    recipients: List[str] = None, upi_ids: List[str] = None,
                    notes: List[str] = None, amounts: List[float] = None) -> np.ndarray:
        """
        Batch embedding for efficiency with enhanced UPI data.
        """
        n = len(texts)
        if modes is None:
            modes = [''] * n
        if recipients is None:
            recipients = [None] * n
        if upi_ids is None:
            upi_ids = [None] * n
        if notes is None:
            notes = [None] * n
        if amounts is None:
            amounts = [None] * n
        
        # Build rich queries
        queries = []
        for i in range(n):
            context_parts = [texts[i] if texts[i] else '']
            
            # Add recipient
            if recipients[i] and not pd.isna(recipients[i]):
                context_parts.append(str(recipients[i]).strip())
            
            # Add note
            if notes[i] and not pd.isna(notes[i]):
                note_str = str(notes[i]).strip()
                if len(note_str) > 2:
                    context_parts.append(note_str)
            
            # Add UPI ID base
            if upi_ids[i] and not pd.isna(upi_ids[i]):
                upi_base = ''.join([c for c in str(upi_ids[i]) if c.isalpha()])
                if len(upi_base) > 3:
                    context_parts.append(upi_base)
            
            # Add mode
            if modes[i]:
                context_parts.append(modes[i])
            
            # Add amount context
            if amounts[i] and amounts[i] > 0:
                if amounts[i] < 100:
                    context_parts.append("small")
                elif amounts[i] < 1000:
                    context_parts.append("medium")
                elif amounts[i] < 10000:
                    context_parts.append("large")
            
            rich_text = ' '.join(context_parts)
            queries.append(f"query: {rich_text}")
        
        # Generate embeddings
        embeddings = self.model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
    
    def clear_cache(self):
        self.cache.clear()
