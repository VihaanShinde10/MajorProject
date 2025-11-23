"""
Cross-Validation Framework for Transaction Categorization.

Implements k-fold cross-validation with temporal awareness.
Ensures robust evaluation and prevents data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Callable
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
import json
from pathlib import Path
from datetime import datetime

class TemporalCrossValidator:
    """
    Cross-validator with temporal awareness.
    Ensures training data always precedes test data (no future leakage).
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate temporal train/test splits.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        n = len(df)
        
        splits = []
        fold_size = n // self.n_splits
        
        for i in range(self.n_splits):
            # Test set: fold i
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            
            # Train set: all data before test set
            train_indices = np.arange(test_start)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits

class TransactionCrossValidator:
    """
    Main cross-validation framework for transaction categorization.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 temporal: bool = True,
                 stratified: bool = False):
        """
        Args:
            n_splits: Number of folds
            temporal: Use temporal splitting (recommended)
            stratified: Use stratified splitting (maintains class balance)
        """
        self.n_splits = n_splits
        self.temporal = temporal
        self.stratified = stratified
        
        if temporal:
            self.splitter = TemporalCrossValidator(n_splits)
        elif stratified:
            self.splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            self.splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def cross_validate(self,
                      df: pd.DataFrame,
                      classifier_fn: Callable,
                      ground_truth_col: str = 'true_category') -> Dict:
        """
        Perform cross-validation.
        
        Args:
            df: DataFrame with transactions
            classifier_fn: Function that takes (train_df, test_df) and returns predictions
            ground_truth_col: Column name for ground truth labels
            
        Returns:
            Cross-validation results
        """
        results = {
            'fold_results': [],
            'overall_metrics': {},
            'confusion_matrices': [],
            'predictions': []
        }
        
        # Generate splits
        if self.temporal:
            splits = self.splitter.split(df)
        elif self.stratified:
            splits = list(self.splitter.split(df, df[ground_truth_col]))
        else:
            splits = list(self.splitter.split(df))
        
        print(f"Running {len(splits)}-fold cross-validation...")
        
        all_predictions = []
        all_ground_truth = []
        all_confidences = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{len(splits)}")
            print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            
            # Split data
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            
            # Run classifier
            predictions, confidences = classifier_fn(train_df, test_df)
            
            # Get ground truth
            ground_truth = test_df[ground_truth_col].tolist()
            
            # Compute fold metrics
            fold_metrics = self._compute_metrics(ground_truth, predictions)
            fold_metrics['fold'] = fold_idx + 1
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['test_size'] = len(test_idx)
            fold_metrics['avg_confidence'] = np.mean(confidences)
            
            results['fold_results'].append(fold_metrics)
            
            # Accumulate for overall metrics
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)
            all_confidences.extend(confidences)
            
            # Confusion matrix
            cm = confusion_matrix(ground_truth, predictions)
            results['confusion_matrices'].append(cm)
            
            print(f"  Precision: {fold_metrics['precision']:.4f}")
            print(f"  Recall: {fold_metrics['recall']:.4f}")
            print(f"  F1-Score: {fold_metrics['f1_score']:.4f}")
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        
        # Compute overall metrics
        results['overall_metrics'] = self._compute_metrics(all_ground_truth, all_predictions)
        results['overall_metrics']['avg_confidence'] = np.mean(all_confidences)
        
        # Aggregate fold statistics
        results['fold_statistics'] = self._aggregate_fold_stats(results['fold_results'])
        
        # Store predictions
        results['predictions'] = {
            'predicted': all_predictions,
            'ground_truth': all_ground_truth,
            'confidences': all_confidences
        }
        
        return results
    
    def _compute_metrics(self, ground_truth: List[str], predictions: List[str]) -> Dict:
        """Compute classification metrics."""
        return {
            'precision': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
            'recall': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(ground_truth, predictions, average='weighted', zero_division=0),
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision_macro': precision_score(ground_truth, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(ground_truth, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(ground_truth, predictions, average='macro', zero_division=0)
        }
    
    def _aggregate_fold_stats(self, fold_results: List[Dict]) -> Dict:
        """Aggregate statistics across folds."""
        metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'avg_confidence']
        stats = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            stats[f'{metric}_min'] = np.min(values)
            stats[f'{metric}_max'] = np.max(values)
        
        return stats
    
    def print_results(self, results: Dict):
        """Print cross-validation results."""
        print("\n" + "="*80)
        print("CROSS-VALIDATION RESULTS")
        print("="*80)
        
        # Fold results
        print("\nPer-Fold Results:")
        print("-"*80)
        print(f"{'Fold':<6} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'Confidence':<12}")
        print("-"*80)
        
        for fold in results['fold_results']:
            print(f"{fold['fold']:<6} "
                  f"{fold['precision']:<12.4f} "
                  f"{fold['recall']:<12.4f} "
                  f"{fold['f1_score']:<12.4f} "
                  f"{fold['accuracy']:<12.4f} "
                  f"{fold['avg_confidence']:<12.4f}")
        
        # Overall results
        print("\n" + "="*80)
        print("OVERALL RESULTS (All Folds Combined)")
        print("="*80)
        overall = results['overall_metrics']
        print(f"Precision (weighted): {overall['precision']:.4f}")
        print(f"Recall (weighted):    {overall['recall']:.4f}")
        print(f"F1-Score (weighted):  {overall['f1_score']:.4f}")
        print(f"Accuracy:             {overall['accuracy']:.4f}")
        print(f"Avg Confidence:       {overall['avg_confidence']:.4f}")
        
        # Fold statistics
        print("\n" + "="*80)
        print("FOLD STATISTICS (Mean ± Std)")
        print("="*80)
        stats = results['fold_statistics']
        print(f"Precision: {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")
        print(f"Recall:    {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}")
        print(f"F1-Score:  {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
        print(f"Accuracy:  {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
        print("="*80)
    
    def save_results(self, results: Dict, output_path: str):
        """Save cross-validation results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        results_copy = results.copy()
        results_copy['confusion_matrices'] = [cm.tolist() for cm in results['confusion_matrices']]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")

def example_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Example classifier function for testing.
    Replace with actual classification pipeline.
    """
    # Dummy implementation - replace with real classifier
    predictions = ['Food & Dining'] * len(test_df)
    confidences = [0.8] * len(test_df)
    return predictions, confidences

if __name__ == '__main__':
    # Example usage
    print("Cross-Validation Framework")
    print("Usage:")
    print("  from evaluation.cross_validation import TransactionCrossValidator")
    print("  ")
    print("  validator = TransactionCrossValidator(n_splits=5, temporal=True)")
    print("  results = validator.cross_validate(df, classifier_fn)")
    print("  validator.print_results(results)")
    print("  validator.save_results(results, 'cv_results.json')")

