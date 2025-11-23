"""
Ensemble Methods and Confidence Calibration.

Combines multiple classifiers and calibrates confidence scores.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import pickle

class EnsembleClassifier:
    """
    Ensemble classifier combining multiple methods.
    """
    
    def __init__(self, calibrate: bool = True):
        """
        Args:
            calibrate: Whether to calibrate confidence scores
        """
        self.calibrate = calibrate
        self.calibrator = None
        self.categories = [
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
    
    def predict_ensemble(self,
                        predictions: List[Tuple[str, float]],
                        method: str = 'weighted_vote') -> Tuple[str, float, Dict]:
        """
        Combine predictions from multiple classifiers.
        
        Args:
            predictions: List of (category, confidence) tuples from different methods
            method: Ensemble method ('weighted_vote', 'max_confidence', 'stacking')
            
        Returns:
            (final_category, final_confidence, provenance)
        """
        if not predictions:
            return 'Others/Uncategorized', 0.0, {'reason': 'No predictions'}
        
        if method == 'weighted_vote':
            return self._weighted_vote(predictions)
        elif method == 'max_confidence':
            return self._max_confidence(predictions)
        elif method == 'stacking':
            return self._stacking(predictions)
        elif method == 'unanimous':
            return self._unanimous_ensemble(predictions)
        else:
            return self._weighted_vote(predictions)
    
    def _weighted_vote(self, predictions: List[Tuple[str, float]]) -> Tuple[str, float, Dict]:
        """Weighted voting based on confidence scores."""
        weighted_votes = {}
        
        for category, confidence in predictions:
            if category not in weighted_votes:
                weighted_votes[category] = 0.0
            weighted_votes[category] += confidence
        
        if not weighted_votes:
            return 'Others/Uncategorized', 0.0, {'reason': 'No valid votes'}
        
        # Get winner
        winner = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(weighted_votes.values())
        final_confidence = weighted_votes[winner] / len(predictions)  # Average confidence
        
        return winner, final_confidence, {
            'method': 'weighted_vote',
            'votes': weighted_votes,
            'n_methods': len(predictions)
        }
    
    def _max_confidence(self, predictions: List[Tuple[str, float]]) -> Tuple[str, float, Dict]:
        """Select prediction with highest confidence."""
        if not predictions:
            return 'Others/Uncategorized', 0.0, {'reason': 'No predictions'}
        
        winner_cat, winner_conf = max(predictions, key=lambda x: x[1])
        
        return winner_cat, winner_conf, {
            'method': 'max_confidence',
            'n_methods': len(predictions)
        }
    
    def _unanimous_ensemble(self, predictions: List[Tuple[str, float]]) -> Tuple[str, float, Dict]:
        """Require unanimous agreement for high confidence."""
        if not predictions:
            return 'Others/Uncategorized', 0.0, {'reason': 'No predictions'}
        
        categories = [cat for cat, conf in predictions]
        confidences = [conf for cat, conf in predictions]
        
        # Check unanimity
        if len(set(categories)) == 1:
            # All agree
            category = categories[0]
            confidence = np.mean(confidences)  # Average confidence
            return category, confidence, {
                'method': 'unanimous',
                'agreement': 'full',
                'n_methods': len(predictions)
            }
        else:
            # No unanimity - fall back to weighted vote
            return self._weighted_vote(predictions)
    
    def _stacking(self, predictions: List[Tuple[str, float]]) -> Tuple[str, float, Dict]:
        """
        Stacking ensemble (meta-learner).
        Requires trained meta-model.
        """
        # TODO: Implement stacking with trained meta-model
        # For now, fall back to weighted vote
        return self._weighted_vote(predictions)
    
    def calibrate_confidence(self, 
                            predictions: List[str],
                            confidences: List[float],
                            ground_truth: List[str],
                            method: str = 'isotonic') -> 'EnsembleClassifier':
        """
        Calibrate confidence scores using ground truth.
        
        Args:
            predictions: Predicted categories
            confidences: Raw confidence scores
            ground_truth: True categories
            method: Calibration method ('isotonic', 'platt')
            
        Returns:
            Self (for chaining)
        """
        # Create binary correctness labels
        is_correct = np.array([pred == truth for pred, truth in zip(predictions, ground_truth)])
        confidences = np.array(confidences)
        
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(confidences, is_correct)
        elif method == 'platt':
            # Platt scaling (logistic regression)
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(confidences.reshape(-1, 1), is_correct)
        
        print(f"✅ Confidence calibrator trained using {method} method")
        return self
    
    def apply_calibration(self, confidence: float) -> float:
        """Apply calibration to a confidence score."""
        if self.calibrator is None:
            return confidence
        
        if isinstance(self.calibrator, IsotonicRegression):
            return self.calibrator.predict([confidence])[0]
        else:
            # Logistic regression
            return self.calibrator.predict_proba([[confidence]])[0, 1]
    
    def save_calibrator(self, path: str):
        """Save calibrator to file."""
        if self.calibrator is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.calibrator, f)
            print(f"✅ Calibrator saved to {path}")
    
    def load_calibrator(self, path: str):
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            self.calibrator = pickle.load(f)
        print(f"✅ Calibrator loaded from {path}")

class ConfidenceAnalyzer:
    """
    Analyze and improve confidence estimation.
    """
    
    def __init__(self):
        self.bins = 10
    
    def compute_calibration_curve(self,
                                  predictions: List[str],
                                  confidences: List[float],
                                  ground_truth: List[str]) -> Dict:
        """
        Compute calibration curve (reliability diagram).
        
        Returns:
            Dictionary with bin statistics
        """
        is_correct = np.array([pred == truth for pred, truth in zip(predictions, ground_truth)])
        confidences = np.array(confidences)
        
        # Create bins
        bin_edges = np.linspace(0, 1, self.bins + 1)
        bin_indices = np.digitize(confidences, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)
        
        # Compute statistics per bin
        bin_stats = []
        for i in range(self.bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = is_correct[mask].mean()
                bin_count = mask.sum()
                
                bin_stats.append({
                    'bin': i,
                    'confidence': bin_conf,
                    'accuracy': bin_acc,
                    'count': int(bin_count),
                    'calibration_error': abs(bin_conf - bin_acc)
                })
        
        # Compute ECE (Expected Calibration Error)
        total_count = len(confidences)
        ece = sum(stat['count'] / total_count * stat['calibration_error'] 
                 for stat in bin_stats)
        
        return {
            'bin_stats': bin_stats,
            'ece': ece,
            'n_samples': total_count
        }
    
    def print_calibration_report(self, calibration_data: Dict):
        """Print calibration analysis report."""
        print("\n" + "="*80)
        print("CONFIDENCE CALIBRATION ANALYSIS")
        print("="*80)
        
        print(f"\nExpected Calibration Error (ECE): {calibration_data['ece']:.4f}")
        print(f"Total Samples: {calibration_data['n_samples']}")
        
        print("\nPer-Bin Statistics:")
        print("-"*80)
        print(f"{'Bin':<6} {'Avg Conf':<12} {'Accuracy':<12} {'Count':<10} {'Cal Error':<12}")
        print("-"*80)
        
        for stat in calibration_data['bin_stats']:
            print(f"{stat['bin']:<6} "
                  f"{stat['confidence']:<12.4f} "
                  f"{stat['accuracy']:<12.4f} "
                  f"{stat['count']:<10} "
                  f"{stat['calibration_error']:<12.4f}")
        
        print("="*80)
        
        # Interpretation
        ece = calibration_data['ece']
        if ece < 0.05:
            print("✅ Excellent calibration (ECE < 0.05)")
        elif ece < 0.10:
            print("✓ Good calibration (ECE < 0.10)")
        elif ece < 0.15:
            print("⚠️ Moderate calibration (ECE < 0.15)")
        else:
            print("❌ Poor calibration (ECE ≥ 0.15) - Consider recalibration")

def create_ensemble_pipeline(rule_result: Tuple,
                             semantic_result: Tuple,
                             behavioral_result: Tuple,
                             zeroshot_result: Tuple = None) -> Tuple[str, float, Dict]:
    """
    Create ensemble from all classification methods.
    
    Args:
        rule_result: (category, confidence, reason) from rule-based
        semantic_result: (category, confidence, provenance) from semantic
        behavioral_result: (category, confidence, provenance) from behavioral
        zeroshot_result: (category, confidence, provenance) from zero-shot
        
    Returns:
        Ensemble prediction
    """
    ensemble = EnsembleClassifier()
    
    predictions = []
    
    # Add rule-based (if available)
    if rule_result[0] is not None:
        predictions.append((rule_result[0], rule_result[1]))
    
    # Add semantic (if available)
    if semantic_result[0] is not None:
        predictions.append((semantic_result[0], semantic_result[1]))
    
    # Add behavioral (if available)
    if behavioral_result[0] is not None:
        predictions.append((behavioral_result[0], behavioral_result[1]))
    
    # Add zero-shot (if available)
    if zeroshot_result and zeroshot_result[0] is not None:
        predictions.append((zeroshot_result[0], zeroshot_result[1]))
    
    # Ensemble prediction
    return ensemble.predict_ensemble(predictions, method='weighted_vote')

if __name__ == '__main__':
    print("Ensemble Classifier and Confidence Calibration")
    print("Usage:")
    print("  from layers.ensemble_classifier import EnsembleClassifier, ConfidenceAnalyzer")
    print("  ensemble = EnsembleClassifier(calibrate=True)")
    print("  category, conf, prov = ensemble.predict_ensemble(predictions)")

