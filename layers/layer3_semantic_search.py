import faiss
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from collections import Counter
import json
import os

class SemanticSearcher:
    def __init__(self, embedding_dim: int = 768, embedder=None, corpus_path: str = None):
        """
        Initialize semantic searcher with category-based matching.
        
        Args:
            embedding_dim: Dimension of embeddings (default: 768 for E5)
            embedder: E5Embedder instance for creating category prototypes
            corpus_path: Path to Mumbai merchants corpus JSON
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
        
        # Load corpus for category prototypes
        if corpus_path is None:
            corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mumbai_merchants_corpus.json')
        
        self.corpus = {}
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
        
        # Build category prototypes if embedder is provided
        if self.embedder and self.corpus:
            self._build_category_prototypes()
    
    def _build_category_prototypes(self):
        """
        Build semantic prototypes for each category from corpus.
        Creates representative embeddings that capture the semantic meaning of each category.
        ROBUST: Handles errors gracefully and validates inputs.
        """
        try:
            print("🔨 Building category prototypes from corpus...")
            
            if not self.embedder:
                print("⚠️ No embedder provided, skipping category prototype building")
                return
            
            if not self.corpus:
                print("⚠️ No corpus loaded, skipping category prototype building")
                return
            
            category_texts = []
            category_names = []
            
            for category in self.fixed_categories:
                if category == 'Others/Uncategorized':
                    # Skip - this is a fallback category
                    continue
                
                if category not in self.corpus:
                    continue
                
                # Create rich textual descriptions for each category
                category_data = self.corpus[category]
                
                # Strategy 1: Combine all subcategory items into descriptive sentences
                descriptive_texts = []
                
                for subcategory, items in category_data.items():
                    if not isinstance(items, list):
                        continue
                    
                    if subcategory == 'keywords':
                        # Use keywords to create context
                        keyword_text = f"{category} includes: {', '.join(items[:10])}"
                        descriptive_texts.append(keyword_text)
                    else:
                        # Create natural language descriptions
                        if len(items) > 0:
                            # Sample top items from each subcategory
                            sample_items = items[:15]  # Take top 15 items
                            desc_text = f"{category} - {subcategory}: {', '.join(sample_items)}"
                            descriptive_texts.append(desc_text)
                
                # Combine all descriptions for this category
                if descriptive_texts:
                    full_category_text = ". ".join(descriptive_texts)
                    category_texts.append(full_category_text)
                    category_names.append(category)
            
            # Generate embeddings for category prototypes
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
    
    def build_index(self, embeddings: np.ndarray, labels: List[str], metadata: List[Dict]):
        """
        Build FAISS index from historical transactions (optional, for additional context).
        This is now secondary to category-based matching.
        
        Args:
            embeddings: Transaction embeddings
            labels: Transaction categories
            metadata: Transaction metadata
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}-dim embeddings")
        
        # Validate and fix categories
        validated_labels = [self._validate_category(label) for label in labels]
        
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        self.transaction_index = faiss.IndexFlatIP(self.embedding_dim)
        self.transaction_index.add(embeddings.astype('float32'))
        self.transaction_labels = validated_labels
        self.transaction_metadata = metadata  # Now includes 'matched_source' from Layer 1
    
    def search(self, query_embedding: np.ndarray, k: int = 20) -> Tuple[Optional[str], float, Dict]:
        """
        Search for best matching category using category prototypes.
        REDESIGNED: Compares against category semantic meanings, not individual transactions.
        
        Returns: (category, confidence, provenance)
        """
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # STRATEGY 1: Category Prototype Matching (PRIMARY)
        # Compare transaction against semantic meaning of each category
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
                    reason = f"Strong category match: {best_category} (sim: {best_sim:.3f}, gap: {gap:.3f})"
                elif best_sim >= 0.70 and gap >= 0.08:
                    confidence = 0.68
                    reason = f"Good category match: {best_category} (sim: {best_sim:.3f}, gap: {gap:.3f})"
                elif best_sim >= 0.65:
                    confidence = 0.60
                    reason = f"Moderate category match: {best_category} (sim: {best_sim:.3f})"
                else:
                    # Low similarity - return None to let other layers handle
                    return None, best_sim, {
                        'method': 'category_prototype_weak',
                        'top_matches': [(cat, float(sim)) for cat, sim in zip(top_categories, top_sims)],
                        'reason': f'Weak category match (best: {best_sim:.3f})'
                    }
            else:
                # Only one category available
                if best_sim >= 0.70:
                    confidence = 0.68
                    reason = f"Category match: {best_category} (sim: {best_sim:.3f})"
                else:
                    return None, best_sim, {
                        'method': 'category_prototype_weak',
                        'top_match': (best_category, best_sim),
                        'reason': f'Weak category match (sim: {best_sim:.3f})'
                    }
            
            return best_category, confidence, {
                'method': 'category_prototype',
                'top_match': (best_category, best_sim),
                'all_matches': [(cat, float(sim)) for cat, sim in zip(top_categories, top_sims)],
                'reason': reason
            }
        
        # STRATEGY 2: Historical Transaction Matching (FALLBACK)
        # Use historical transactions if category prototypes not available
        if self.transaction_index is None or self.transaction_index.ntotal == 0:
            return None, 0.0, {'reason': 'No category prototypes or historical transactions available'}
        
        # Search top-k historical transactions
        k_search = min(k, self.transaction_index.ntotal)
        similarities, indices = self.transaction_index.search(query_embedding, k_search)
        
        # Get labels for top matches
        top_labels = [self.transaction_labels[idx] for idx in indices[0]]
        top_sims = similarities[0]
        
        # Fallback: Use historical transaction consensus (less reliable than category prototypes)
        # Strategy 1: Unanimous top-3 with HIGH similarity
        if len(top_labels) >= 3:
            top3_labels = top_labels[:3]
            top3_sims = top_sims[:3]
            top3_indices = indices[0][:3]
            
            # All 3 must match AND have high similarity
            if len(set(top3_labels)) == 1 and top3_sims[0] >= 0.85 and top3_sims[2] >= 0.75:
                # Check if top match has UPI field metadata
                top_metadata = self.transaction_metadata[top3_indices[0]] if top3_indices[0] < len(self.transaction_metadata) else {}
                matched_source = top_metadata.get('matched_source', 'unknown')
                
                return top3_labels[0], 0.65, {
                    'method': 'historical_unanimous_top3',
                    'top_match': (top3_labels[0], float(top3_sims[0])),
                    'matches': [(label, float(sim)) for label, sim in zip(top3_labels, top3_sims)],
                    'matched_source': matched_source,
                    'reason': f'Historical: Top 3 unanimous ({top3_sims[0]:.3f} similarity)'
                }
        
        # Strategy 2: Strong majority in top-10
        if len(top_labels) >= 10:
            top10_labels = top_labels[:10]
            top10_sims = top_sims[:10]
            
            label_counts = Counter(top10_labels)
            most_common_label, count = label_counts.most_common(1)[0]
            
            # At least 7/10 match AND first result has good similarity
            if count >= 7 and top10_sims[0] >= 0.75:
                matching_sims = [top10_sims[i] for i, lbl in enumerate(top10_labels) if lbl == most_common_label]
                avg_sim = np.mean(matching_sims)
                
                if avg_sim >= 0.68:
                    confidence = 0.60 if count >= 8 else 0.55
                    
                    return most_common_label, confidence, {
                        'method': 'historical_majority_top10',
                        'top_match': (most_common_label, float(top10_sims[0])),
                        'matches': [(label, float(sim)) for label, sim in zip(top10_labels, top10_sims)],
                        'majority_count': count,
                        'avg_similarity': float(avg_sim),
                        'reason': f'Historical: Majority ({count}/10, avg sim: {avg_sim:.3f})'
                    }
        
        # No consensus - return None (let other layers handle it)
        return None, float(top_sims[0]) if len(top_sims) > 0 else 0.0, {
            'method': 'no_consensus',
            'top_match': (top_labels[0], float(top_sims[0])) if len(top_labels) > 0 else ('None', 0.0),
            'matches': [(label, float(sim)) for label, sim in zip(top_labels[:5], top_sims[:5])],
            'reason': f'No consensus (top similarity: {top_sims[0]:.3f})'
        }
    
    @property
    def index(self):
        """Backward compatibility: return transaction index."""
        return self.transaction_index
    
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
