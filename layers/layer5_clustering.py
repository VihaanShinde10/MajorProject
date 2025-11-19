import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import hdbscan
from typing import Tuple, Optional, Dict, List
from collections import Counter

class BehavioralClusterer:
    def __init__(self):
        self.clusterer = None
        self.scaler = StandardScaler()
        self.cluster_labels = {}
        self.feature_vectors = None
        self.transaction_ids = []
        self.categories = []
        
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
    
    def fit(self, features_df: pd.DataFrame, categories: List[str] = None):
        """Fit HDBSCAN on behavioral features."""
        # Handle NaN values - fill with 0 or median
        features_clean = features_df.copy()
        
        # Check for NaN, inf, or missing values
        if features_clean.isnull().any().any():
            print("Warning: NaN values detected in features, filling with 0")
            features_clean = features_clean.fillna(0)
        
        # Replace inf values
        features_clean = features_clean.replace([np.inf, -np.inf], 0)
        
        # Ensure we have valid numeric data
        features_clean = features_clean.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        n_samples = len(features_clean)
        
        # IMPROVED: Dynamic parameters based on dataset size
        # Smaller min_cluster_size = More clusters (more granular)
        # Larger min_cluster_size = Fewer clusters (more general)
        if n_samples < 50:
            min_cluster_size = 3
            min_samples = 2
        elif n_samples < 100:
            min_cluster_size = 5
            min_samples = 3
        elif n_samples < 500:
            min_cluster_size = max(3, int(0.03 * n_samples))  # 3% of data
            min_samples = max(2, int(0.01 * n_samples))       # 1% of data
        else:
            min_cluster_size = max(5, int(0.02 * n_samples))  # 2% of data
            min_samples = max(3, int(0.008 * n_samples))      # 0.8% of data
        
        print(f"Clustering with {n_samples} transactions: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        # Standardize features
        try:
            features_scaled = self.scaler.fit_transform(features_clean)
        except Exception as e:
            print(f"Error in scaling: {e}")
            # Fallback: use unscaled features
            features_scaled = features_clean.values
        
        # Final NaN check after scaling
        if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
            print("Warning: NaN/Inf detected after scaling, replacing with 0")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_vectors = features_scaled
        
        # Fit HDBSCAN with improved parameters
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.1,  # Reduced from 0.3 for more clusters
            cluster_selection_method='eom',  # Excess of Mass (better for varied densities)
            prediction_data=True
        )
        
        cluster_ids = self.clusterer.fit_predict(features_scaled)
        
        # Label clusters using semantic categories if provided
        if categories is not None:
            self._label_clusters(cluster_ids, categories)
        
        return cluster_ids
    
    def predict(self, features: Dict[str, float], semantic_category: Optional[str] = None) -> Tuple[Optional[str], float, Dict]:
        """
        Predict category for new transaction using clustering.
        Returns: (category, confidence, provenance)
        """
        if self.clusterer is None:
            return None, 0.0, {'reason': 'Clusterer not fitted'}
        
        # Convert features to array and handle NaN/inf
        feature_values = list(features.values())
        feature_array = np.array(feature_values, dtype=float).reshape(1, -1)
        
        # Replace NaN and inf values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            feature_scaled = self.scaler.transform(feature_array)
            # Final check after scaling
            feature_scaled = np.nan_to_num(feature_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"Error in scaling prediction features: {e}")
            return None, 0.0, {'reason': f'Scaling error: {str(e)}'}
        
        # Get cluster assignment with soft membership
        cluster_id, membership_prob = self._soft_predict(feature_scaled[0])
        
        if cluster_id == -1:  # Noise point
            return None, 0.0, {
                'method': 'cluster_noise',
                'cluster_id': -1,
                'reason': 'Classified as noise by HDBSCAN'
            }
        
        # Get cluster label
        cluster_category = self.cluster_labels.get(cluster_id)
        
        if cluster_category is None:
            return None, membership_prob, {
                'method': 'unlabeled_cluster',
                'cluster_id': int(cluster_id),
                'membership_prob': float(membership_prob),
                'reason': 'Cluster has no semantic label'
            }
        
        # KNN refinement within cluster
        knn_category, knn_confidence = self._knn_refinement(feature_scaled[0], cluster_id)
        
        if knn_category:
            final_confidence = membership_prob * knn_confidence
            return knn_category, final_confidence, {
                'method': 'hdbscan_knn',
                'cluster_id': int(cluster_id),
                'cluster_label': cluster_category,
                'membership_prob': float(membership_prob),
                'knn_confidence': float(knn_confidence),
                'reason': f'Behavioral cluster match (cluster {cluster_id})'
            }
        
        return cluster_category, membership_prob * 0.9, {
            'method': 'hdbscan_only',
            'cluster_id': int(cluster_id),
            'membership_prob': float(membership_prob),
            'reason': f'Cluster membership (cluster {cluster_id})'
        }
    
    def _soft_predict(self, feature_vector: np.ndarray) -> Tuple[int, float]:
        """Predict cluster with soft membership probability."""
        try:
            cluster_id, membership_probs = hdbscan.approximate_predict(self.clusterer, feature_vector.reshape(1, -1))
            
            # Handle different return types from hdbscan
            # cluster_id can be array or scalar
            if isinstance(cluster_id, np.ndarray):
                cluster_id_val = int(cluster_id[0])
            else:
                cluster_id_val = int(cluster_id)
            
            # membership_probs can be array or scalar
            if isinstance(membership_probs, np.ndarray):
                if membership_probs.ndim > 1 and len(membership_probs[0]) > 0:
                    max_prob = float(np.max(membership_probs[0]))
                elif membership_probs.ndim == 1 and len(membership_probs) > 0:
                    max_prob = float(np.max(membership_probs))
                else:
                    max_prob = 0.6
            elif isinstance(membership_probs, (float, np.floating)):
                max_prob = float(membership_probs)
            else:
                max_prob = 0.6
            
            return cluster_id_val, max_prob
            
        except Exception as e:
            print(f"Warning: Error in soft predict: {e}")
            # Fallback: use hard prediction
            try:
                cluster_id = self.clusterer.predict([feature_vector])[0]
                return int(cluster_id), 0.6
            except:
                return -1, 0.0
    
    def _label_clusters(self, cluster_ids: np.ndarray, categories: List[str]):
        """Assign semantic labels to clusters by majority vote."""
        unique_clusters = set(cluster_ids)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue
            
            # Get categories for transactions in this cluster
            mask = cluster_ids == cluster_id
            cluster_categories = [cat for cat, m in zip(categories, mask) if m and cat]
            
            if cluster_categories:
                # Majority vote
                most_common = Counter(cluster_categories).most_common(1)[0][0]
                # Validate category
                validated_category = self._validate_category(most_common)
                self.cluster_labels[cluster_id] = validated_category
    
    def _knn_refinement(self, feature_vector: np.ndarray, cluster_id: int, k: int = 5) -> Tuple[Optional[str], float]:
        """Refine prediction using KNN within cluster."""
        if self.feature_vectors is None or len(self.categories) == 0:
            return None, 0.0
        
        # Get points in same cluster
        cluster_mask = self.clusterer.labels_ == cluster_id
        cluster_features = self.feature_vectors[cluster_mask]
        cluster_categories = [cat for cat, m in zip(self.categories, cluster_mask) if m]
        
        if len(cluster_features) < k:
            return None, 0.0
        
        # Find k nearest neighbors
        knn = NearestNeighbors(n_neighbors=min(k, len(cluster_features)))
        knn.fit(cluster_features)
        
        distances, indices = knn.kneighbors(feature_vector.reshape(1, -1))
        
        # Get categories of neighbors
        neighbor_categories = [cluster_categories[idx] for idx in indices[0]]
        
        # Majority vote
        if neighbor_categories:
            category_counts = Counter(neighbor_categories)
            most_common, count = category_counts.most_common(1)[0]
            
            majority_fraction = count / len(neighbor_categories)
            
            if majority_fraction >= 0.7:
                # Validate category
                validated_category = self._validate_category(most_common)
                return validated_category, majority_fraction
        
        return None, 0.0
    
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

