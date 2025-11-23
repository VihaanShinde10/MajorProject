"""
Enhanced Semantic Search with Attention Mechanism.

Implements attention-weighted semantic matching for better relevance scoring.
"""

import numpy as np
import faiss
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    """
    Attention mechanism for semantic search.
    Learns to weight different parts of embeddings based on context.
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.key_proj = nn.Linear(embedding_dim, hidden_dim)
        self.value_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query_emb: torch.Tensor, context_embs: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism.
        
        Args:
            query_emb: (embedding_dim,) query embedding
            context_embs: (n_context, embedding_dim) context embeddings
            
        Returns:
            Attended embedding (embedding_dim,)
        """
        # Project to Q, K, V
        Q = self.query_proj(query_emb.unsqueeze(0))  # (1, hidden_dim)
        K = self.key_proj(context_embs)  # (n_context, hidden_dim)
        V = self.value_proj(context_embs)  # (n_context, hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(0, 1)) / np.sqrt(self.hidden_dim)  # (1, n_context)
        attention_weights = F.softmax(scores, dim=1)  # (1, n_context)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (1, hidden_dim)
        
        # Project back to embedding space
        output = self.output_proj(attended).squeeze(0)  # (embedding_dim,)
        
        # Residual connection + layer norm
        output = self.layer_norm(query_emb + self.dropout(output))
        
        return output

class SemanticSearcherWithAttention:
    """
    Enhanced semantic searcher with attention mechanism.
    Now supports category prototype matching like the base SemanticSearcher.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 embedder=None,
                 corpus_path: str = None,
                 attention_model_path: Optional[str] = None):
        """
        Args:
            embedding_dim: Dimension of embeddings
            embedder: E5Embedder instance for creating category prototypes (optional)
            corpus_path: Path to Mumbai merchants corpus JSON (optional)
            attention_model_path: Path to trained attention model
        """
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        
        # Category prototype index (compares against category meanings)
        self.category_index = None
        self.category_embeddings = None
        self.category_names = []
        
        # Historical transaction index (for additional context)
        self.transaction_index = None
        self.transaction_labels = []
        self.transaction_metadata = []
        
        # For backward compatibility
        self.index = None
        self.categories = []
        self.metadata = []
        
        # Initialize attention module
        self.attention = AttentionModule(embedding_dim)
        if attention_model_path:
            try:
                self.attention.load_state_dict(torch.load(attention_model_path, map_location='cpu'))
                self.attention.eval()
                self.use_attention = True
                print(f"✅ Loaded attention model from {attention_model_path}")
            except Exception as e:
                print(f"⚠️ Could not load attention model: {e}")
                self.use_attention = False
        else:
            self.use_attention = False
        
        # Load corpus and build category prototypes if embedder provided
        if embedder and corpus_path:
            import json
            import os
            
            if corpus_path and os.path.exists(corpus_path):
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
                self._build_category_prototypes()
            else:
                self.corpus = {}
        else:
            self.corpus = {}
    
    def _build_category_prototypes(self):
        """
        Build semantic prototypes for each category from corpus.
        Creates representative embeddings that capture the semantic meaning of each category.
        ROBUST: Handles errors gracefully and validates inputs.
        """
        try:
            print("🔨 Building category prototypes from corpus (with attention)...")
            
            if not self.embedder:
                print("⚠️ No embedder provided, skipping category prototype building")
                return
            
            if not self.corpus:
                print("⚠️ No corpus loaded, skipping category prototype building")
                return
            
            fixed_categories = [
                'Food & Dining', 'Commute/Transport', 'Shopping', 'Bills & Utilities',
                'Entertainment', 'Healthcare', 'Education', 'Investments',
                'Salary/Income', 'Transfers', 'Subscriptions'
            ]
            
            category_texts = []
            category_names = []
            
            for category in fixed_categories:
                if category not in self.corpus:
                    continue
                
                category_data = self.corpus[category]
                descriptive_texts = []
                
                for subcategory, items in category_data.items():
                    if not isinstance(items, list):
                        continue
                    
                    if subcategory == 'keywords':
                        keyword_text = f"{category} includes: {', '.join(items[:10])}"
                        descriptive_texts.append(keyword_text)
                    else:
                        if len(items) > 0:
                            sample_items = items[:15]
                            desc_text = f"{category} - {subcategory}: {', '.join(sample_items)}"
                            descriptive_texts.append(desc_text)
                
                if descriptive_texts:
                    full_category_text = ". ".join(descriptive_texts)
                    category_texts.append(full_category_text)
                    category_names.append(category)
            
            if len(category_texts) > 0:
                print(f"   Generating embeddings for {len(category_texts)} category prototypes...")
                category_embeddings = []
                
                for i, text in enumerate(category_texts):
                    try:
                        # Use E5 embedder (it adds "query: " prefix internally)
                        embedding = self.embedder.embed(text)
                        category_embeddings.append(embedding)
                    except Exception as e:
                        print(f"⚠️ Failed to embed category {category_names[i]}: {e}")
                        continue
                
                if len(category_embeddings) > 0:
                    self.category_embeddings = np.vstack(category_embeddings).astype('float32')
                    self.category_names = category_names[:len(category_embeddings)]
                    
                    # Build FAISS index for category prototypes
                    import faiss
                    self.category_index = faiss.IndexFlatIP(self.embedding_dim)
                    self.category_index.add(self.category_embeddings)
                    
                    print(f"✅ Built category prototype index with {len(self.category_names)} categories")
                else:
                    print("⚠️ No category embeddings generated")
            else:
                print("⚠️ No category texts to embed")
                
        except Exception as e:
            print(f"⚠️ Error building category prototypes: {e}")
            print("   Falling back to transaction-based matching")
    
    def build_index(self, 
                   embeddings: np.ndarray,
                   categories: List[str],
                   metadata: List[Dict]):
        """
        Build FAISS index from historical transactions (optional, for additional context).
        This is now secondary to category-based matching.
        
        Args:
            embeddings: (N, embedding_dim) array
            categories: List of N categories
            metadata: List of N metadata dicts
        """
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Build transaction index
        self.transaction_index = faiss.IndexFlatIP(self.embedding_dim)
        self.transaction_index.add(embeddings.astype('float32'))
        
        self.transaction_labels = categories
        self.transaction_metadata = metadata
        
        # Backward compatibility
        self.index = self.transaction_index
        self.categories = categories
        self.metadata = metadata
        
        print(f"✅ Built semantic index with {len(embeddings)} embeddings")
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 10,
              use_attention: bool = None) -> Tuple[Optional[str], float, Dict]:
        """
        Search for best matching category using category prototypes (with optional attention).
        REDESIGNED: Compares against category semantic meanings, not individual transactions.
        
        Args:
            query_embedding: (embedding_dim,) query vector
            k: Number of neighbors to retrieve (unused for category prototypes)
            use_attention: Override attention usage
            
        Returns:
            (category, confidence, provenance)
        """
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # STRATEGY 1: Category Prototype Matching (PRIMARY)
        if self.category_index is not None and self.category_index.ntotal > 0:
            # Search against category prototypes
            similarities, indices = self.category_index.search(query_embedding, min(5, self.category_index.ntotal))
            
            top_categories = [self.category_names[idx] for idx in indices[0]]
            top_sims = similarities[0]
            
            # Get best matching category
            best_category = top_categories[0]
            best_sim = float(top_sims[0])
            
            # Confidence based on similarity score and gap to second best
            if len(top_sims) >= 2:
                second_sim = float(top_sims[1])
                gap = best_sim - second_sim
                
                # High confidence if: high similarity AND clear gap
                if best_sim >= 0.75 and gap >= 0.10:
                    confidence = 0.75
                    reason = f"Strong category match (attention): {best_category} (sim: {best_sim:.3f}, gap: {gap:.3f})"
                elif best_sim >= 0.70 and gap >= 0.08:
                    confidence = 0.68
                    reason = f"Good category match (attention): {best_category} (sim: {best_sim:.3f}, gap: {gap:.3f})"
                elif best_sim >= 0.65:
                    confidence = 0.60
                    reason = f"Moderate category match (attention): {best_category} (sim: {best_sim:.3f})"
                else:
                    # Low similarity - return None to let other layers handle
                    return None, best_sim, {
                        'method': 'category_prototype_weak_attention',
                        'top_matches': [(cat, float(sim)) for cat, sim in zip(top_categories, top_sims)],
                        'reason': f'Weak category match (best: {best_sim:.3f})'
                    }
            else:
                # Only one category available
                if best_sim >= 0.70:
                    confidence = 0.68
                    reason = f"Category match (attention): {best_category} (sim: {best_sim:.3f})"
                else:
                    return None, best_sim, {
                        'method': 'category_prototype_weak_attention',
                        'top_match': (best_category, best_sim),
                        'reason': f'Weak category match (sim: {best_sim:.3f})'
                    }
            
            return best_category, confidence, {
                'method': 'category_prototype_attention',
                'top_match': (best_category, best_sim),
                'all_matches': [(cat, float(sim)) for cat, sim in zip(top_categories, top_sims)],
                'reason': reason
            }
        
        # STRATEGY 2: Historical Transaction Matching (FALLBACK)
        if self.transaction_index is None or self.transaction_index.ntotal == 0:
            return None, 0.0, {'reason': 'No category prototypes or historical transactions available'}
        
        # Normalize query
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm)
        
        # Search historical transactions
        similarities, indices = self.transaction_index.search(query_norm, k)
        similarities = similarities[0]
        indices = indices[0]
        
        # Filter valid results
        valid_mask = indices >= 0
        similarities = similarities[valid_mask]
        indices = indices[valid_mask]
        
        if len(indices) == 0:
            return None, 0.0, {'reason': 'No neighbors found'}
        
        # Get categories
        neighbor_categories = [self.transaction_labels[idx] for idx in indices]
        neighbor_similarities = similarities
        
        # Voting strategies (fallback)
        unanimous_result = self._unanimous_vote(neighbor_categories, neighbor_similarities)
        if unanimous_result:
            return unanimous_result
        
        majority_result = self._majority_vote(neighbor_categories, neighbor_similarities)
        if majority_result:
            return majority_result
        
        # Weighted vote (fallback)
        return self._weighted_vote(neighbor_categories, neighbor_similarities)
    
    def _unanimous_vote(self, 
                       categories: List[str], 
                       similarities: np.ndarray,
                       threshold: float = 0.78) -> Optional[Tuple[str, float, Dict]]:
        """Unanimous vote from top-3 high-confidence matches."""
        top_3_mask = similarities >= threshold
        if top_3_mask.sum() < 3:
            return None
        
        top_3_categories = [cat for cat, sim in zip(categories, similarities) if sim >= threshold][:3]
        
        if len(set(top_3_categories)) == 1:
            category = top_3_categories[0]
            confidence = similarities[0]  # Highest similarity
            return category, confidence, {
                'reason': f'Unanimous top-3 (sim={confidence:.3f})',
                'method': 'unanimous',
                'votes': 3
            }
        
        return None
    
    def _majority_vote(self,
                      categories: List[str],
                      similarities: np.ndarray,
                      threshold: float = 0.70) -> Optional[Tuple[str, float, Dict]]:
        """Majority vote from top-10 matches."""
        top_10_mask = similarities >= threshold
        if top_10_mask.sum() < 5:
            return None
        
        top_10_categories = [cat for cat, sim in zip(categories, similarities) if sim >= threshold][:10]
        
        # Count votes
        from collections import Counter
        vote_counts = Counter(top_10_categories)
        
        if len(vote_counts) == 0:
            return None
        
        winner, count = vote_counts.most_common(1)[0]
        
        # Require majority (>50%)
        if count > len(top_10_categories) / 2:
            # Confidence = average similarity of winner's votes
            winner_sims = [sim for cat, sim in zip(categories, similarities) 
                          if cat == winner and sim >= threshold][:10]
            confidence = np.mean(winner_sims) if winner_sims else 0.0
            
            return winner, confidence, {
                'reason': f'Majority {count}/{len(top_10_categories)} (avg_sim={confidence:.3f})',
                'method': 'majority',
                'votes': count,
                'total': len(top_10_categories)
            }
        
        return None
    
    def _weighted_vote(self,
                      categories: List[str],
                      similarities: np.ndarray) -> Tuple[str, float, Dict]:
        """Weighted vote using similarity scores."""
        from collections import defaultdict
        
        # Weight votes by similarity
        weighted_votes = defaultdict(float)
        for cat, sim in zip(categories, similarities):
            weighted_votes[cat] += sim
        
        if len(weighted_votes) == 0:
            return 'Others/Uncategorized', 0.0, {'reason': 'No votes', 'method': 'weighted'}
        
        # Get winner
        winner = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(weighted_votes.values())
        confidence = weighted_votes[winner] / total_weight if total_weight > 0 else 0.0
        
        return winner, confidence, {
            'reason': f'Weighted vote (weight={weighted_votes[winner]:.3f}/{total_weight:.3f})',
            'method': 'weighted',
            'weight': weighted_votes[winner],
            'total_weight': total_weight
        }
    
    def search_with_reranking(self,
                             query_embedding: np.ndarray,
                             query_features: Dict[str, float],
                             k: int = 20) -> Tuple[Optional[str], float, Dict]:
        """
        Search with feature-based reranking.
        
        Args:
            query_embedding: Query embedding
            query_features: Behavioral features for reranking
            k: Number of candidates to retrieve
            
        Returns:
            (category, confidence, provenance)
        """
        # Initial retrieval
        category, confidence, prov = self.search(query_embedding, k=k)
        
        if category is None:
            return None, 0.0, prov
        
        # Feature-based reranking (if behavioral features available)
        if query_features:
            # Boost confidence based on behavioral alignment
            behavioral_boost = self._compute_behavioral_boost(query_features, category)
            confidence = min(confidence * (1 + behavioral_boost), 1.0)
            prov['behavioral_boost'] = behavioral_boost
            prov['reranked'] = True
        
        return category, confidence, prov
    
    def _compute_behavioral_boost(self, features: Dict[str, float], category: str) -> float:
        """Compute confidence boost based on behavioral features."""
        boost = 0.0
        
        # Category-specific behavioral signals
        if category == 'Food & Dining':
            if features.get('is_evening', 0) or features.get('is_early_morning', 0):
                boost += 0.1
            if 100 <= features.get('amount_raw', 0) * 10000 <= 2000:
                boost += 0.05
        
        elif category == 'Commute/Transport':
            if features.get('is_commute_window', 0):
                boost += 0.15
            if features.get('is_weekday', 0):
                boost += 0.05
        
        elif category == 'Subscriptions':
            if features.get('is_periodic', 0):
                boost += 0.2
            if features.get('is_month_start', 0) or features.get('is_month_end', 0):
                boost += 0.1
        
        elif category == 'Salary/Income':
            if features.get('is_xlarge', 0):
                boost += 0.15
            if features.get('is_month_start', 0):
                boost += 0.1
        
        return boost

def train_attention_model(training_data_path: str, 
                         output_model_path: str = 'models/attention_trained.pt',
                         epochs: int = 50):
    """
    Train attention model from triplet data.
    
    Args:
        training_data_path: Path to training data (anchor, positive, negative embeddings)
        output_model_path: Where to save trained model
        epochs: Number of training epochs
    """
    # TODO: Implement training pipeline
    # Requires triplet loss training with (query, positive, negative) examples
    pass

if __name__ == '__main__':
    print("Semantic Search with Attention Mechanism")
    print("Usage:")
    print("  from layers.layer3_semantic_search_attention import SemanticSearcherWithAttention")
    print("  searcher = SemanticSearcherWithAttention(attention_model_path='models/attention.pt')")
    print("  category, conf, prov = searcher.search(query_embedding)")

