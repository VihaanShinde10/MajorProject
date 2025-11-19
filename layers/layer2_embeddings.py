import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pandas as pd

class E5Embedder:
    def __init__(self, model_name: str = 'intfloat/e5-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 768
        self.cache = {}
    
    def embed(self, text: str, transaction_mode: str = '', description: str = '') -> np.ndarray:
        """
        Generate E5 embedding for transaction text.
        Returns: 768-dim L2-normalized vector
        """
        if not text:
            return np.zeros(self.embedding_dim)
        
        # Check cache
        cache_key = f"{text}_{transaction_mode}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # E5 requires "query: " prefix
        query = f"query: {text} {transaction_mode} {description}".strip()
        
        # Generate embedding
        embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Cache result
        self.cache[cache_key] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str], modes: List[str] = None) -> np.ndarray:
        """Batch embedding for efficiency."""
        if modes is None:
            modes = [''] * len(texts)
        
        queries = [f"query: {t} {m}".strip() for t, m in zip(texts, modes)]
        embeddings = self.model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
    
    def clear_cache(self):
        self.cache.clear()

