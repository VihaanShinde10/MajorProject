"""
Advanced Metrics and Visualization for Transaction Categorization.

Implements comprehensive evaluation metrics beyond basic precision/recall.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class AdvancedMetricsCalculator:
    """
    Calculate advanced metrics for transaction categorization.
    """
    
    def __init__(self, categories: List[str]):
        self.categories = categories
    
    def compute_all_metrics(self,
                           predictions: List[str],
                           ground_truth: List[str],
                           confidences: List[float]) -> Dict:
        """
        Compute comprehensive metrics.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['basic'] = self._compute_basic_metrics(predictions, ground_truth)
        
        # Per-category metrics
        metrics['per_category'] = self._compute_per_category_metrics(predictions, ground_truth)
        
        # Confidence metrics
        metrics['confidence'] = self._compute_confidence_metrics(
            predictions, ground_truth, confidences
        )
        
        # Agreement metrics
        metrics['agreement'] = self._compute_agreement_metrics(predictions, ground_truth)
        
        # Cost-sensitive metrics
        metrics['cost_sensitive'] = self._compute_cost_sensitive_metrics(
            predictions, ground_truth
        )
        
        # Confusion analysis
        metrics['confusion'] = self._analyze_confusion(predictions, ground_truth)
        
        return metrics
    
    def _compute_basic_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Compute basic classification metrics."""
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted', zero_division=0
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='macro', zero_division=0
        )
        
        return {
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'accuracy': np.mean([p == g for p, g in zip(predictions, ground_truth)])
        }
    
    def _compute_per_category_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Compute metrics for each category."""
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, labels=self.categories, zero_division=0
        )
        
        per_category = {}
        for i, category in enumerate(self.categories):
            per_category[category] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }
        
        return per_category
    
    def _compute_confidence_metrics(self,
                                    predictions: List[str],
                                    ground_truth: List[str],
                                    confidences: List[float]) -> Dict:
        """Compute confidence-related metrics."""
        is_correct = np.array([p == g for p, g in zip(predictions, ground_truth)])
        confidences = np.array(confidences)
        
        # Confidence statistics
        correct_confs = confidences[is_correct]
        incorrect_confs = confidences[~is_correct]
        
        metrics = {
            'avg_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'avg_confidence_correct': float(np.mean(correct_confs)) if len(correct_confs) > 0 else 0.0,
            'avg_confidence_incorrect': float(np.mean(incorrect_confs)) if len(incorrect_confs) > 0 else 0.0,
            'confidence_separation': float(np.mean(correct_confs) - np.mean(incorrect_confs)) if len(correct_confs) > 0 and len(incorrect_confs) > 0 else 0.0
        }
        
        # Brier score (calibration metric)
        metrics['brier_score'] = float(np.mean((confidences - is_correct) ** 2))
        
        # Confidence thresholds analysis
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high_conf_mask = confidences >= threshold
            if high_conf_mask.sum() > 0:
                metrics[f'accuracy_at_{threshold}'] = float(is_correct[high_conf_mask].mean())
                metrics[f'coverage_at_{threshold}'] = float(high_conf_mask.mean())
        
        return metrics
    
    def _compute_agreement_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Compute inter-rater agreement metrics."""
        return {
            'cohen_kappa': cohen_kappa_score(ground_truth, predictions),
            'matthews_corrcoef': matthews_corrcoef(ground_truth, predictions)
        }
    
    def _compute_cost_sensitive_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """
        Compute cost-sensitive metrics.
        Different misclassifications have different costs.
        """
        # Define cost matrix (example: misclassifying salary as food is worse than food as dining)
        cost_matrix = self._get_cost_matrix()
        
        total_cost = 0.0
        n = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            if pred != truth:
                cost = cost_matrix.get((truth, pred), 1.0)
                total_cost += cost
        
        return {
            'total_cost': total_cost,
            'avg_cost_per_error': total_cost / max(1, n - sum(p == g for p, g in zip(predictions, ground_truth))),
            'normalized_cost': total_cost / n
        }
    
    def _get_cost_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Define misclassification costs.
        Higher cost = more severe error.
        """
        # Critical misclassifications (high cost)
        high_cost = [
            ('Salary/Income', 'Food & Dining'),  # Salary as food is bad
            ('Salary/Income', 'Shopping'),
            ('Investments', 'Entertainment'),  # Investment as entertainment is bad
            ('Bills & Utilities', 'Entertainment'),
        ]
        
        # Moderate misclassifications
        moderate_cost = [
            ('Food & Dining', 'Shopping'),  # Similar categories
            ('Commute/Transport', 'Shopping'),
        ]
        
        cost_matrix = {}
        
        # High cost errors
        for pair in high_cost:
            cost_matrix[pair] = 3.0
        
        # Moderate cost errors
        for pair in moderate_cost:
            cost_matrix[pair] = 1.5
        
        # Default cost for other errors
        # (will use 1.0 as default in compute_cost_sensitive_metrics)
        
        return cost_matrix
    
    def _analyze_confusion(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Analyze confusion patterns."""
        cm = confusion_matrix(ground_truth, predictions, labels=self.categories)
        
        # Find most confused pairs
        confused_pairs = []
        for i, true_cat in enumerate(self.categories):
            for j, pred_cat in enumerate(self.categories):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_category': true_cat,
                        'predicted_category': pred_cat,
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100) if cm[i].sum() > 0 else 0.0
                    })
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'top_confusions': confused_pairs[:10]  # Top 10 confusion pairs
        }
    
    def print_detailed_report(self, metrics: Dict):
        """Print comprehensive metrics report."""
        print("\n" + "="*80)
        print("ADVANCED METRICS REPORT")
        print("="*80)
        
        # Basic metrics
        print("\n📊 BASIC METRICS")
        print("-"*80)
        basic = metrics['basic']
        print(f"Accuracy:           {basic['accuracy']:.4f}")
        print(f"Precision (weighted): {basic['precision_weighted']:.4f}")
        print(f"Recall (weighted):    {basic['recall_weighted']:.4f}")
        print(f"F1-Score (weighted):  {basic['f1_weighted']:.4f}")
        print(f"Precision (macro):    {basic['precision_macro']:.4f}")
        print(f"Recall (macro):       {basic['recall_macro']:.4f}")
        print(f"F1-Score (macro):     {basic['f1_macro']:.4f}")
        
        # Per-category metrics
        print("\n📈 PER-CATEGORY METRICS")
        print("-"*80)
        print(f"{'Category':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*80)
        for category, cat_metrics in metrics['per_category'].items():
            print(f"{category:<25} "
                  f"{cat_metrics['precision']:<12.4f} "
                  f"{cat_metrics['recall']:<12.4f} "
                  f"{cat_metrics['f1_score']:<12.4f} "
                  f"{cat_metrics['support']:<10}")
        
        # Confidence metrics
        print("\n🎯 CONFIDENCE METRICS")
        print("-"*80)
        conf = metrics['confidence']
        print(f"Average Confidence:              {conf['avg_confidence']:.4f}")
        print(f"Std Dev Confidence:              {conf['std_confidence']:.4f}")
        print(f"Avg Confidence (Correct):        {conf['avg_confidence_correct']:.4f}")
        print(f"Avg Confidence (Incorrect):      {conf['avg_confidence_incorrect']:.4f}")
        print(f"Confidence Separation:           {conf['confidence_separation']:.4f}")
        print(f"Brier Score (lower=better):      {conf['brier_score']:.4f}")
        
        print("\nAccuracy at Confidence Thresholds:")
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if f'accuracy_at_{threshold}' in conf:
                acc = conf[f'accuracy_at_{threshold}']
                cov = conf[f'coverage_at_{threshold}']
                print(f"  ≥{threshold}: Accuracy={acc:.4f}, Coverage={cov:.2%}")
        
        # Agreement metrics
        print("\n🤝 AGREEMENT METRICS")
        print("-"*80)
        agree = metrics['agreement']
        print(f"Cohen's Kappa:        {agree['cohen_kappa']:.4f}")
        print(f"Matthews Corr Coef:   {agree['matthews_corrcoef']:.4f}")
        
        # Cost-sensitive metrics
        print("\n💰 COST-SENSITIVE METRICS")
        print("-"*80)
        cost = metrics['cost_sensitive']
        print(f"Total Cost:           {cost['total_cost']:.2f}")
        print(f"Avg Cost per Error:   {cost['avg_cost_per_error']:.2f}")
        print(f"Normalized Cost:      {cost['normalized_cost']:.4f}")
        
        # Top confusions
        print("\n🔀 TOP CONFUSION PAIRS")
        print("-"*80)
        print(f"{'True Category':<25} {'→ Predicted As':<25} {'Count':<10} {'%':<10}")
        print("-"*80)
        for confusion in metrics['confusion']['top_confusions'][:5]:
            print(f"{confusion['true_category']:<25} "
                  f"→ {confusion['predicted_category']:<25} "
                  f"{confusion['count']:<10} "
                  f"{confusion['percentage']:<10.2f}%")
        
        print("="*80)
    
    def plot_confusion_matrix(self, 
                             predictions: List[str],
                             ground_truth: List[str],
                             output_path: str = 'confusion_matrix.png'):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(ground_truth, predictions, labels=self.categories)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories)
        plt.title('Confusion Matrix')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved to {output_path}")
    
    def plot_confidence_distribution(self,
                                    predictions: List[str],
                                    ground_truth: List[str],
                                    confidences: List[float],
                                    output_path: str = 'confidence_dist.png'):
        """Plot confidence distribution for correct vs incorrect predictions."""
        is_correct = np.array([p == g for p, g in zip(predictions, ground_truth)])
        confidences = np.array(confidences)
        
        plt.figure(figsize=(10, 6))
        
        # Correct predictions
        plt.hist(confidences[is_correct], bins=20, alpha=0.6, label='Correct', color='green')
        
        # Incorrect predictions
        plt.hist(confidences[~is_correct], bins=20, alpha=0.6, label='Incorrect', color='red')
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confidence distribution saved to {output_path}")

if __name__ == '__main__':
    print("Advanced Metrics Calculator")
    print("Usage:")
    print("  from evaluation.advanced_metrics import AdvancedMetricsCalculator")
    print("  calculator = AdvancedMetricsCalculator(categories)")
    print("  metrics = calculator.compute_all_metrics(predictions, ground_truth, confidences)")
    print("  calculator.print_detailed_report(metrics)")

