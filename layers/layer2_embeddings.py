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
        ROBUST: Prioritizes recipient_name, upi_id, and note with HIGHER WEIGHT.
        
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
        
        # ROBUST STRATEGY: Build context with UPI fields FIRST and REPEATED for higher weight
        context_parts = []
        
        # Priority 1: Add RECIPIENT_NAME first (most reliable) - TRIPLE WEIGHT
        if recipient_name and not pd.isna(recipient_name):
            recipient_clean = str(recipient_name).strip()
            if recipient_clean and recipient_clean not in ['', 'nan', 'none']:
                # Add 3 times for higher semantic weight
                context_parts.extend([recipient_clean, recipient_clean, recipient_clean])
        
        # Priority 2: Add NOTE (provides category hints) - DOUBLE WEIGHT
        if note and not pd.isna(note):
            note_clean = str(note).strip()
            if note_clean and len(note_clean) > 2 and note_clean not in ['', 'nan', 'none', 'upi']:
                # Add twice for higher semantic weight
                context_parts.extend([note_clean, note_clean])
        
        # Priority 3: Add UPI_ID (contains merchant info) - DOUBLE WEIGHT
        if upi_id and not pd.isna(upi_id):
            upi_clean = str(upi_id).strip()
            if upi_clean and upi_clean not in ['', 'nan', 'none']:
                # Extract meaningful parts (e.g., "paytmqr" from "paytmqr1jc")
                upi_base = ''.join([c for c in upi_clean if c.isalpha()])
                if upi_base and len(upi_base) > 3:
                    # Add twice for higher semantic weight
                    context_parts.extend([upi_base, upi_base])
        
        # Priority 4: Add normalized text (fallback)
        if text:
            context_parts.append(text)
        
        # Priority 5: Add transaction mode
        if transaction_mode:
            context_parts.append(transaction_mode)
        
        # Priority 6: Add amount range (for context: small/medium/large transaction)
        if amount and amount > 0:
            if amount < 100:
                context_parts.append("small payment")
            elif amount < 1000:
                context_parts.append("medium transaction")
            elif amount < 10000:
                context_parts.append("large payment")
            else:
                context_parts.append("very large transaction")
        
        # Combine all context (UPI fields now have 2-3x weight)
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
        ROBUST: Prioritizes recipient_name, upi_id, and note with HIGHER WEIGHT.
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
        
        # Build rich queries with UPI fields FIRST and WEIGHTED
        queries = []
        for i in range(n):
            context_parts = []
            
            # Priority 1: Add RECIPIENT (TRIPLE WEIGHT)
            if recipients[i] and not pd.isna(recipients[i]):
                recipient_clean = str(recipients[i]).strip()
                if recipient_clean and recipient_clean not in ['', 'nan', 'none']:
                    context_parts.extend([recipient_clean, recipient_clean, recipient_clean])
            
            # Priority 2: Add NOTE (DOUBLE WEIGHT)
            if notes[i] and not pd.isna(notes[i]):
                note_str = str(notes[i]).strip()
                if len(note_str) > 2 and note_str not in ['', 'nan', 'none', 'upi']:
                    context_parts.extend([note_str, note_str])
            
            # Priority 3: Add UPI ID base (DOUBLE WEIGHT)
            if upi_ids[i] and not pd.isna(upi_ids[i]):
                upi_clean = str(upi_ids[i]).strip()
                if upi_clean and upi_clean not in ['', 'nan', 'none']:
                    upi_base = ''.join([c for c in upi_clean if c.isalpha()])
                    if len(upi_base) > 3:
                        context_parts.extend([upi_base, upi_base])
            
            # Priority 4: Add normalized text
            if texts[i]:
                context_parts.append(texts[i])
            
            # Priority 5: Add mode
            if modes[i]:
                context_parts.append(modes[i])
            
            # Priority 6: Add amount context
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
