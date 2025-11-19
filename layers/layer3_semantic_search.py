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
        """
        Build FAISS index from embeddings.
        ROBUST: Stores metadata about UPI field matching for better provenance.
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}-dim embeddings")
        
        # Validate and fix categories
        validated_labels = [self._validate_category(label) for label in labels]
        
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.labels = validated_labels
        self.metadata = metadata  # Now includes 'matched_source' from Layer 1
    
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
        
        # Strategy 1: Unanimous top-3 with VERY HIGH similarity (ultra-strict)
        # Goal: Only match when we have VERY strong semantic agreement
        if len(top_labels) >= 3:
            top3_labels = top_labels[:3]
            top3_sims = top_sims[:3]
            top3_indices = indices[0][:3]
            
            # All 3 must match AND have VERY high similarity (increased thresholds)
            if len(set(top3_labels)) == 1 and top3_sims[0] >= 0.90 and top3_sims[2] >= 0.82:  # Increased from 0.85, 0.75
                # Check if top match has UPI field metadata
                top_metadata = self.metadata[top3_indices[0]] if top3_indices[0] < len(self.metadata) else {}
                matched_source = top_metadata.get('matched_source', 'unknown')
                
                return top3_labels[0], 0.95, {
                    'method': 'unanimous_top3_strict',
                    'top_match': (top3_labels[0], float(top3_sims[0])),
                    'matches': [(label, float(sim)) for label, sim in zip(top3_labels, top3_sims)],
                    'matched_source': matched_source,  # NEW: Track UPI field match
                    'reason': f'Top 3 unanimous ({top3_sims[0]:.3f} similarity, source: {matched_source})'
                }
        
        # Strategy 2: Strong majority in top-10 (7+ out of 10) - STRICTER
        # Goal: Require stronger consensus before committing to a category
        if len(top_labels) >= 10:
            top10_labels = top_labels[:10]
            top10_sims = top_sims[:10]
            top10_indices = indices[0][:10]
            
            label_counts = Counter(top10_labels)
            most_common_label, count = label_counts.most_common(1)[0]
            
            # At least 7/10 match AND first result has good similarity (increased thresholds)
            if count >= 7 and top10_sims[0] >= 0.80:  # Increased from 6 and 0.75
                # Calculate average similarity for matching labels
                matching_sims = [top10_sims[i] for i, lbl in enumerate(top10_labels) if lbl == most_common_label]
                avg_sim = np.mean(matching_sims)
                
                # Require higher average similarity too
                if avg_sim >= 0.72:  # New requirement
                    confidence = 0.82 if count >= 8 else 0.73  # Adjusted
                    
                    # Check top match metadata
                    top_metadata = self.metadata[top10_indices[0]] if top10_indices[0] < len(self.metadata) else {}
                    matched_source = top_metadata.get('matched_source', 'unknown')
                    
                    return most_common_label, confidence, {
                        'method': 'majority_top10',
                        'top_match': (most_common_label, float(top10_sims[0])),
                        'matches': [(label, float(sim)) for label, sim in zip(top10_labels, top10_sims)],
                        'majority_count': count,
                        'avg_similarity': float(avg_sim),
                        'matched_source': matched_source,  # NEW: Track UPI field match
                        'reason': f'Strong majority ({count}/10, avg sim: {avg_sim:.3f}, source: {matched_source})'
                    }
        
        # Strategy 3: Super strong top-5 majority - STRICTER
        # Goal: Only accept when top 5 have very strong agreement
        if len(top_labels) >= 5:
            top5_labels = top_labels[:5]
            top5_sims = top_sims[:5]
            top5_indices = indices[0][:5]
            
            label_counts = Counter(top5_labels)
            most_common_label, count = label_counts.most_common(1)[0]
            
            # 4 or 5 out of 5 match with HIGHER similarity threshold
            if count >= 4 and top5_sims[0] >= 0.85:  # Increased from 0.80
                avg_sim = np.mean([top5_sims[i] for i, lbl in enumerate(top5_labels) if lbl == most_common_label])
                
                # Require higher average too
                if avg_sim >= 0.78:  # New requirement
                    confidence = 0.88 if count == 5 else 0.78  # Adjusted
                    
                    # Check top match metadata
                    top_metadata = self.metadata[top5_indices[0]] if top5_indices[0] < len(self.metadata) else {}
                    matched_source = top_metadata.get('matched_source', 'unknown')
                    
                    return most_common_label, confidence, {
                        'method': 'strong_top5',
                        'top_match': (most_common_label, float(top5_sims[0])),
                        'matches': [(label, float(sim)) for label, sim in zip(top5_labels, top5_sims)],
                        'majority_count': count,
                        'avg_similarity': float(avg_sim),
                        'matched_source': matched_source,  # NEW: Track UPI field match
                        'reason': f'Strong top-5 consensus ({count}/5, avg sim: {avg_sim:.3f}, source: {matched_source})'
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
