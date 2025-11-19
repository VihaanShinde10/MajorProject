import faiss
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from collections import Counter

class SemanticSearcher:
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = None
        self.labels = []
        self.metadata = []
    
    def build_index(self, embeddings: np.ndarray, labels: List[str], metadata: List[Dict]):
        """Build FAISS index from embeddings."""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}-dim embeddings")
        
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.labels = labels
        self.metadata = metadata
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[Optional[str], float, Dict]:
        """
        Search for similar transactions.
        Returns: (category, confidence, provenance)
        """
        if self.index is None or self.index.ntotal == 0:
            return None, 0.0, {'reason': 'Empty index'}
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search top-k
        similarities, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Get labels for top matches
        top_labels = [self.labels[idx] for idx in indices[0]]
        top_sims = similarities[0]
        
        # Check unanimous top-3
        if len(top_labels) >= 3:
            top3_labels = top_labels[:3]
            top3_sims = top_sims[:3]
            
            if len(set(top3_labels)) == 1 and top3_sims[0] >= 0.78:
                return top3_labels[0], 0.9 * top3_sims[0], {
                    'method': 'unanimous_top3',
                    'matches': list(zip(top3_labels, top3_sims.tolist())),
                    'reason': 'Top 3 unanimous with high similarity'
                }
        
        # Check majority in top-10
        label_counts = Counter(top_labels)
        most_common_label, count = label_counts.most_common(1)[0]
        
        if count >= 6 and top_sims[0] >= 0.70:
            avg_sim = np.mean([top_sims[i] for i, lbl in enumerate(top_labels) if lbl == most_common_label])
            return most_common_label, 0.75 * avg_sim, {
                'method': 'majority_top10',
                'matches': list(zip(top_labels, top_sims.tolist())),
                'majority_count': count,
                'reason': f'Majority ({count}/10) with acceptable similarity'
            }
        
        return None, float(top_sims[0]) if len(top_sims) > 0 else 0.0, {
            'method': 'no_consensus',
            'matches': list(zip(top_labels, top_sims.tolist())),
            'reason': 'No consensus in semantic search'
        }

