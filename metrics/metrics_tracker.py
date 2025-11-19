import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score
import json

class MetricsTracker:
    """
    Metrics tracker for UNSUPERVISED transaction categorization.
    Tracks operational metrics without requiring ground truth labels.
    """
    def __init__(self):
        self.predictions = []
        self.confidences = []
        self.layers_used = []
        self.timestamps = []
        self.correction_history = []
        self.alpha_values = []  # Gating weights
        self.merchants = []
        
        # Clustering metrics
        self.feature_vectors = None
        self.cluster_labels = None
        self.true_labels = None  # Optional, for V-measure
        
    def log_prediction(self, 
                      predicted_category: str,
                      confidence: float,
                      layer_used: str,
                      alpha: float = None,
                      merchant: str = None,
                      true_category: str = None):
        """Log a prediction for metrics tracking."""
        self.predictions.append(predicted_category)
        self.confidences.append(confidence)
        self.layers_used.append(layer_used)
        self.timestamps.append(datetime.now())
        
        if alpha is not None:
            self.alpha_values.append(alpha)
        if merchant:
            self.merchants.append(merchant)
        if true_category:
            if self.true_labels is None:
                self.true_labels = []
            self.true_labels.append(true_category)
    
    def set_clustering_data(self, 
                           feature_vectors: np.ndarray,
                           cluster_labels: np.ndarray):
        """
        Store clustering data for quality metrics.
        
        Args:
            feature_vectors: Standardized feature vectors (n_samples, n_features)
            cluster_labels: Cluster assignments (n_samples,)
        """
        self.feature_vectors = feature_vectors
        self.cluster_labels = cluster_labels
    
    def log_correction(self, 
                      original_prediction: str,
                      corrected_category: str,
                      confidence: float):
        """Log user correction - key metric for unsupervised systems."""
        self.correction_history.append({
            'timestamp': datetime.now(),
            'original': original_prediction,
            'corrected': corrected_category,
            'confidence': confidence
        })
    
    def compute_metrics(self) -> Dict:
        """Compute unsupervised metrics (no ground truth needed)."""
        if len(self.predictions) == 0:
            return {}
        
        confidences_array = np.array(self.confidences)
        
        # Core operational metrics
        metrics = {
            'total_transactions': len(self.predictions),
            'unique_categories': len(set(self.predictions)),
            
            # Confidence metrics
            'avg_confidence': float(np.mean(confidences_array)),
            'median_confidence': float(np.median(confidences_array)),
            'std_confidence': float(np.std(confidences_array)),
            'min_confidence': float(np.min(confidences_array)),
            'max_confidence': float(np.max(confidences_array)),
            
            # Auto-label metrics (key for unsupervised)
            'auto_label_rate': float(sum(1 for c in self.confidences if c >= 0.75) / len(self.confidences)),
            'probable_rate': float(sum(1 for c in self.confidences if 0.50 <= c < 0.75) / len(self.confidences)),
            'low_confidence_rate': float(sum(1 for c in self.confidences if c < 0.50) / len(self.confidences)),
            
            # Confidence distribution
            'confidence_percentiles': {
                'p25': float(np.percentile(confidences_array, 25)),
                'p50': float(np.percentile(confidences_array, 50)),
                'p75': float(np.percentile(confidences_array, 75)),
                'p90': float(np.percentile(confidences_array, 90)),
                'p95': float(np.percentile(confidences_array, 95))
            }
        }
        
        # Category distribution
        category_counts = Counter(self.predictions)
        metrics['category_distribution'] = dict(category_counts)
        metrics['category_percentages'] = {
            cat: float(count / len(self.predictions) * 100)
            for cat, count in category_counts.items()
        }
        
        # Layer performance (which layer is being used most)
        layer_counts = Counter(self.layers_used)
        metrics['layer_distribution'] = dict(layer_counts)
        metrics['layer_percentages'] = {
            layer: float(count / len(self.layers_used) * 100)
            for layer, count in layer_counts.items()
        }
        
        # Layer-wise confidence
        layer_confidence = {}
        for layer in set(self.layers_used):
            layer_confs = [c for c, l in zip(self.confidences, self.layers_used) if l == layer]
            if layer_confs:
                layer_confidence[layer] = {
                    'avg_confidence': float(np.mean(layer_confs)),
                    'count': len(layer_confs)
                }
        metrics['layer_confidence'] = layer_confidence
        
        # Gating metrics (α distribution)
        if self.alpha_values:
            alpha_array = np.array(self.alpha_values)
            metrics['gating_stats'] = {
                'avg_alpha': float(np.mean(alpha_array)),
                'median_alpha': float(np.median(alpha_array)),
                'text_dominant_rate': float(sum(1 for a in self.alpha_values if a >= 0.5) / len(self.alpha_values)),
                'behavior_dominant_rate': float(sum(1 for a in self.alpha_values if a < 0.5) / len(self.alpha_values))
            }
        
        # Merchant entropy (how consistent is categorization per merchant)
        if self.merchants:
            merchant_categories = {}
            for merchant, category in zip(self.merchants, self.predictions):
                if merchant not in merchant_categories:
                    merchant_categories[merchant] = []
                merchant_categories[merchant].append(category)
            
            # Calculate entropy per merchant
            merchant_consistency = []
            for merchant, cats in merchant_categories.items():
                if len(cats) > 1:
                    most_common_count = Counter(cats).most_common(1)[0][1]
                    consistency = most_common_count / len(cats)
                    merchant_consistency.append(consistency)
            
            if merchant_consistency:
                metrics['merchant_consistency'] = {
                    'avg_consistency': float(np.mean(merchant_consistency)),
                    'merchants_tracked': len(merchant_categories)
                }
        
        # Correction rate (if corrections made)
        if len(self.correction_history) > 0:
            metrics['correction_rate'] = float(len(self.correction_history) / len(self.predictions))
            
            # Which categories get corrected most
            correction_from = Counter([c['original'] for c in self.correction_history])
            metrics['most_corrected_categories'] = dict(correction_from.most_common(5))
        else:
            metrics['correction_rate'] = 0.0
        
        # Category-wise confidence
        category_confidence = {}
        for cat in set(self.predictions):
            cat_confs = [c for c, p in zip(self.confidences, self.predictions) if p == cat]
            if cat_confs:
                category_confidence[cat] = {
                    'avg_confidence': float(np.mean(cat_confs)),
                    'count': len(cat_confs),
                    'auto_label_rate': float(sum(1 for c in cat_confs if c >= 0.75) / len(cat_confs))
                }
        metrics['category_confidence'] = category_confidence
        
        # Temporal metrics
        if len(self.timestamps) > 1:
            time_diffs = [(self.timestamps[i+1] - self.timestamps[i]).total_seconds() 
                         for i in range(len(self.timestamps)-1)]
            metrics['processing_stats'] = {
                'avg_time_per_txn': float(np.mean(time_diffs)),
                'total_processing_time': float(sum(time_diffs))
            }
        
        # Clustering quality metrics
        clustering_metrics = self._compute_clustering_metrics()
        if clustering_metrics:
            metrics['clustering_quality'] = clustering_metrics
        
        return metrics
    
    def _compute_clustering_metrics(self) -> Optional[Dict]:
        """
        Compute clustering quality metrics.
        
        Returns dict with:
        - silhouette_score: Higher is better (range: -1 to 1)
        - davies_bouldin_index: Lower is better (0 to ∞)
        - v_measure: If true labels available (0 to 1)
        """
        if self.feature_vectors is None or self.cluster_labels is None:
            return None
        
        # Filter out noise points (cluster_label = -1)
        valid_mask = self.cluster_labels != -1
        valid_features = self.feature_vectors[valid_mask]
        valid_labels = self.cluster_labels[valid_mask]
        
        if len(valid_features) < 2 or len(set(valid_labels)) < 2:
            return None
        
        metrics = {}
        
        try:
            # Silhouette Score (higher is better, range: -1 to 1)
            # Measures how similar objects are to their own cluster vs other clusters
            silhouette = silhouette_score(valid_features, valid_labels, metric='euclidean')
            metrics['silhouette_score'] = float(silhouette)
            
            # Davies-Bouldin Index (lower is better, 0 to ∞)
            # Measures average similarity ratio of each cluster with most similar one
            db_index = davies_bouldin_score(valid_features, valid_labels)
            metrics['davies_bouldin_index'] = float(db_index)
            
            # V-measure (if true labels available, 0 to 1)
            # Harmonic mean of homogeneity and completeness
            if self.true_labels is not None and len(self.true_labels) == len(self.cluster_labels):
                valid_true_labels = [self.true_labels[i] for i, m in enumerate(valid_mask) if m]
                v_measure = v_measure_score(valid_true_labels, valid_labels)
                metrics['v_measure'] = float(v_measure)
            
            # Additional cluster statistics
            unique_clusters = len(set(valid_labels))
            noise_points = np.sum(self.cluster_labels == -1)
            
            metrics['n_clusters'] = int(unique_clusters)
            metrics['n_noise_points'] = int(noise_points)
            metrics['noise_ratio'] = float(noise_points / len(self.cluster_labels))
            
            # Cluster sizes
            cluster_sizes = Counter(valid_labels)
            metrics['avg_cluster_size'] = float(np.mean(list(cluster_sizes.values())))
            metrics['min_cluster_size'] = int(min(cluster_sizes.values()))
            metrics['max_cluster_size'] = int(max(cluster_sizes.values()))
            
        except Exception as e:
            # If clustering metrics fail, return None
            return None
        
        return metrics
    
    def get_category_summary(self) -> pd.DataFrame:
        """Get summary statistics per category."""
        if not self.predictions:
            return pd.DataFrame()
        
        summary = []
        for cat in set(self.predictions):
            cat_confs = [c for c, p in zip(self.confidences, self.predictions) if p == cat]
            summary.append({
                'Category': cat,
                'Count': len(cat_confs),
                'Percentage': len(cat_confs) / len(self.predictions) * 100,
                'Avg Confidence': np.mean(cat_confs),
                'Min Confidence': np.min(cat_confs),
                'Max Confidence': np.max(cat_confs),
                'Auto-Label Rate': sum(1 for c in cat_confs if c >= 0.75) / len(cat_confs) * 100
            })
        
        return pd.DataFrame(summary).sort_values('Count', ascending=False)
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        metrics = self.compute_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def reset(self):
        """Reset all tracking."""
        self.predictions = []
        self.confidences = []
        self.layers_used = []
        self.timestamps = []
        self.correction_history = []
        self.alpha_values = []
        self.merchants = []
        self.feature_vectors = None
        self.cluster_labels = None
        self.true_labels = None

