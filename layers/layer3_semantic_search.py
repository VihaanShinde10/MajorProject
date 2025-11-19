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
        
        # Fixed categories - enforce these
        self.fixed_categories = [
            'Food & Dining',
            'Commute/Transport',
            'Shopping',
            'Bills & Utilities',
            'Entertainment',
            'Healthcare',
            'Education',
            'Investments',
            'Salary/Income',
            'Transfers',
            'Subscriptions',
            'Others/Uncategorized'
        ]
    
    def build_index(self, embeddings: np.ndarray, labels: List[str], metadata: List[Dict]):
        """Build FAISS index from embeddings."""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}-dim embeddings")
        
        # Validate and fix categories
        validated_labels = [self._validate_category(label) for label in labels]
        
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.labels = validated_labels
        self.metadata = metadata
    
    def search(self, query_embedding: np.ndarray, k: int = 20) -> Tuple[Optional[str], float, Dict]:
        """
        Search for similar transactions with improved thresholds.
        Returns: (category, confidence, provenance)
        """
        if self.index is None or self.index.ntotal == 0:
            return None, 0.0, {'reason': 'Empty index'}
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search top-k (increased from 10 to 20 for better consensus)
        k_search = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, k_search)
        
        # Get labels for top matches
        top_labels = [self.labels[idx] for idx in indices[0]]
        top_sims = similarities[0]
        
        # Strategy 1: Unanimous top-3 with HIGH similarity (strictest)
        if len(top_labels) >= 3:
            top3_labels = top_labels[:3]
            top3_sims = top_sims[:3]
            
            # All 3 must match AND have high similarity
            if len(set(top3_labels)) == 1 and top3_sims[0] >= 0.85 and top3_sims[2] >= 0.75:
                return top3_labels[0], 0.95, {
                    'method': 'unanimous_top3_strict',
                    'top_match': (top3_labels[0], float(top3_sims[0])),
                    'matches': [(label, float(sim)) for label, sim in zip(top3_labels, top3_sims)],
                    'reason': f'Top 3 unanimous ({top3_sims[0]:.3f} similarity)'
                }
        
        # Strategy 2: Strong majority in top-10 (6+ out of 10)
        if len(top_labels) >= 10:
            top10_labels = top_labels[:10]
            top10_sims = top_sims[:10]
            
            label_counts = Counter(top10_labels)
            most_common_label, count = label_counts.most_common(1)[0]
            
            # At least 6/10 match AND first result is reasonably similar
            if count >= 6 and top10_sims[0] >= 0.75:
                # Calculate average similarity for matching labels
                matching_sims = [top10_sims[i] for i, lbl in enumerate(top10_labels) if lbl == most_common_label]
                avg_sim = np.mean(matching_sims)
                
                confidence = 0.80 if count >= 7 else 0.70
                
                return most_common_label, confidence, {
                    'method': 'majority_top10',
                    'top_match': (most_common_label, float(top10_sims[0])),
                    'matches': [(label, float(sim)) for label, sim in zip(top10_labels, top10_sims)],
                    'majority_count': count,
                    'avg_similarity': float(avg_sim),
                    'reason': f'Strong majority ({count}/10, avg sim: {avg_sim:.3f})'
                }
        
        # Strategy 3: Super strong top-5 majority
        if len(top_labels) >= 5:
            top5_labels = top_labels[:5]
            top5_sims = top_sims[:5]
            
            label_counts = Counter(top5_labels)
            most_common_label, count = label_counts.most_common(1)[0]
            
            # 4 or 5 out of 5 match with good similarity
            if count >= 4 and top5_sims[0] >= 0.80:
                avg_sim = np.mean([top5_sims[i] for i, lbl in enumerate(top5_labels) if lbl == most_common_label])
                
                confidence = 0.85 if count == 5 else 0.75
                
                return most_common_label, confidence, {
                    'method': 'strong_top5',
                    'top_match': (most_common_label, float(top5_sims[0])),
                    'matches': [(label, float(sim)) for label, sim in zip(top5_labels, top5_sims)],
                    'majority_count': count,
                    'avg_similarity': float(avg_sim),
                    'reason': f'Strong top-5 consensus ({count}/5, avg sim: {avg_sim:.3f})'
                }
        
        # No consensus - return None (let other layers handle it)
        return None, float(top_sims[0]) if len(top_sims) > 0 else 0.0, {
            'method': 'no_consensus',
            'top_match': (top_labels[0], float(top_sims[0])) if len(top_labels) > 0 else ('None', 0.0),
            'matches': [(label, float(sim)) for label, sim in zip(top_labels[:5], top_sims[:5])],
            'reason': f'No consensus (top similarity: {top_sims[0]:.3f})'
        }
    
    def _validate_category(self, category: str) -> str:
        """Ensure category is in fixed list."""
        if category in self.fixed_categories:
            return category
        
        # Try to map similar categories
        category_lower = category.lower()
        for fixed_cat in self.fixed_categories:
            if category_lower in fixed_cat.lower() or fixed_cat.lower() in category_lower:
                return fixed_cat
        
        # Default to Others
        return 'Others/Uncategorized'
