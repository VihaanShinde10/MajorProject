"""
Ablation Study Baselines for Transaction Categorization.

Implements baseline methods for comparison:
1. Semantic-only (text-based)
2. Behavioral-only (pattern-based)
3. Fixed 50-50 fusion
4. Adaptive fusion (our method)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

@dataclass
class BaselineResult:
    """Results from a baseline method."""
    name: str
    predictions: List[str]
    confidences: List[float]
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    avg_confidence: float

class AblationBaselines:
    """
    Implements baseline methods for ablation study.
    """
    
    def __init__(self):
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
    
    def semantic_only_baseline(self, 
                               semantic_predictions: List[str],
                               semantic_confidences: List[float],
                               ground_truth: List[str]) -> BaselineResult:
        """
        Baseline 1: Semantic-only (text-based classification).
        Uses only text embeddings and semantic search.
        """
        # Use semantic predictions directly
        predictions = []
        confidences = []
        
        for pred, conf in zip(semantic_predictions, semantic_confidences):
            if pred and conf > 0.3:  # Minimum threshold
                predictions.append(pred)
                confidences.append(conf)
            else:
                predictions.append('Others/Uncategorized')
                confidences.append(0.0)
        
        # Compute metrics
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(ground_truth, predictions)
        avg_conf = np.mean(confidences)
        
        return BaselineResult(
            name='Semantic-Only',
            predictions=predictions,
            confidences=confidences,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            avg_confidence=avg_conf
        )
    
    def behavioral_only_baseline(self,
                                 behavioral_predictions: List[str],
                                 behavioral_confidences: List[float],
                                 ground_truth: List[str]) -> BaselineResult:
        """
        Baseline 2: Behavioral-only (pattern-based classification).
        Uses only behavioral features and clustering.
        """
        # Use behavioral predictions directly
        predictions = []
        confidences = []
        
        for pred, conf in zip(behavioral_predictions, behavioral_confidences):
            if pred and conf > 0.3:  # Minimum threshold
                predictions.append(pred)
                confidences.append(conf)
            else:
                predictions.append('Others/Uncategorized')
                confidences.append(0.0)
        
        # Compute metrics
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(ground_truth, predictions)
        avg_conf = np.mean(confidences)
        
        return BaselineResult(
            name='Behavioral-Only',
            predictions=predictions,
            confidences=confidences,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            avg_confidence=avg_conf
        )
    
    def fixed_fusion_baseline(self,
                             semantic_predictions: List[str],
                             semantic_confidences: List[float],
                             behavioral_predictions: List[str],
                             behavioral_confidences: List[float],
                             ground_truth: List[str],
                             alpha: float = 0.5) -> BaselineResult:
        """
        Baseline 3: Fixed 50-50 fusion (or any fixed alpha).
        Combines semantic and behavioral with fixed weight.
        
        Args:
            alpha: Fixed weight for semantic (0.5 = 50-50)
        """
        predictions = []
        confidences = []
        
        for sem_pred, sem_conf, beh_pred, beh_conf in zip(
            semantic_predictions, semantic_confidences,
            behavioral_predictions, behavioral_confidences
        ):
            # Fuse confidences
            fused_conf = alpha * sem_conf + (1 - alpha) * beh_conf
            
            # Choose prediction based on alpha
            if alpha > 0.5:
                final_pred = sem_pred if sem_pred else beh_pred
            else:
                final_pred = beh_pred if beh_pred else sem_pred
            
            if not final_pred:
                final_pred = 'Others/Uncategorized'
                fused_conf = 0.0
            
            predictions.append(final_pred)
            confidences.append(fused_conf)
        
        # Compute metrics
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(ground_truth, predictions)
        avg_conf = np.mean(confidences)
        
        return BaselineResult(
            name=f'Fixed-Fusion (α={alpha})',
            predictions=predictions,
            confidences=confidences,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            avg_confidence=avg_conf
        )
    
    def adaptive_fusion_baseline(self,
                                 semantic_predictions: List[str],
                                 semantic_confidences: List[float],
                                 behavioral_predictions: List[str],
                                 behavioral_confidences: List[float],
                                 alphas: List[float],
                                 ground_truth: List[str]) -> BaselineResult:
        """
        Baseline 4: Adaptive fusion (our method).
        Uses learned/heuristic gating to dynamically weight methods.
        """
        predictions = []
        confidences = []
        
        for sem_pred, sem_conf, beh_pred, beh_conf, alpha in zip(
            semantic_predictions, semantic_confidences,
            behavioral_predictions, behavioral_confidences,
            alphas
        ):
            # Fuse confidences with adaptive alpha
            fused_conf = alpha * sem_conf + (1 - alpha) * beh_conf
            
            # Choose prediction based on alpha
            if alpha > 0.5:
                final_pred = sem_pred if sem_pred else beh_pred
            else:
                final_pred = beh_pred if beh_pred else sem_pred
            
            if not final_pred:
                final_pred = 'Others/Uncategorized'
                fused_conf = 0.0
            
            predictions.append(final_pred)
            confidences.append(fused_conf)
        
        # Compute metrics
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(ground_truth, predictions)
        avg_conf = np.mean(confidences)
        
        return BaselineResult(
            name='Adaptive-Fusion (Ours)',
            predictions=predictions,
            confidences=confidences,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            avg_confidence=avg_conf
        )
    
    def run_ablation_study(self,
                          semantic_predictions: List[str],
                          semantic_confidences: List[float],
                          behavioral_predictions: List[str],
                          behavioral_confidences: List[float],
                          alphas: List[float],
                          ground_truth: List[str]) -> Dict[str, BaselineResult]:
        """
        Run complete ablation study with all baselines.
        
        Returns:
            Dictionary mapping baseline name to results
        """
        results = {}
        
        # Baseline 1: Semantic-only
        print("Running Semantic-Only baseline...")
        results['semantic_only'] = self.semantic_only_baseline(
            semantic_predictions, semantic_confidences, ground_truth
        )
        
        # Baseline 2: Behavioral-only
        print("Running Behavioral-Only baseline...")
        results['behavioral_only'] = self.behavioral_only_baseline(
            behavioral_predictions, behavioral_confidences, ground_truth
        )
        
        # Baseline 3: Fixed 50-50 fusion
        print("Running Fixed 50-50 Fusion baseline...")
        results['fixed_50_50'] = self.fixed_fusion_baseline(
            semantic_predictions, semantic_confidences,
            behavioral_predictions, behavioral_confidences,
            ground_truth, alpha=0.5
        )
        
        # Baseline 3b: Fixed 70-30 fusion (text-heavy)
        print("Running Fixed 70-30 Fusion baseline...")
        results['fixed_70_30'] = self.fixed_fusion_baseline(
            semantic_predictions, semantic_confidences,
            behavioral_predictions, behavioral_confidences,
            ground_truth, alpha=0.7
        )
        
        # Baseline 4: Adaptive fusion (our method)
        print("Running Adaptive Fusion (Ours)...")
        results['adaptive_fusion'] = self.adaptive_fusion_baseline(
            semantic_predictions, semantic_confidences,
            behavioral_predictions, behavioral_confidences,
            alphas, ground_truth
        )
        
        return results
    
    def print_comparison_table(self, results: Dict[str, BaselineResult]):
        """Print comparison table of all baselines."""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)
        print(f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'Avg Conf':<12}")
        print("-"*80)
        
        for name, result in results.items():
            print(f"{result.name:<25} "
                  f"{result.precision:<12.4f} "
                  f"{result.recall:<12.4f} "
                  f"{result.f1_score:<12.4f} "
                  f"{result.accuracy:<12.4f} "
                  f"{result.avg_confidence:<12.4f}")
        
        print("="*80)
        
        # Find best method
        best_f1 = max(results.values(), key=lambda x: x.f1_score)
        print(f"\n🏆 Best F1-Score: {best_f1.name} ({best_f1.f1_score:.4f})")
        
        # Compute improvement
        if 'adaptive_fusion' in results and 'semantic_only' in results:
            improvement = (results['adaptive_fusion'].f1_score - 
                          results['semantic_only'].f1_score)
            print(f"📈 Improvement over Semantic-Only: {improvement:+.4f} ({improvement/results['semantic_only'].f1_score*100:+.2f}%)")
        
        print()
    
    def save_results(self, results: Dict[str, BaselineResult], output_path: str):
        """Save ablation study results to CSV."""
        rows = []
        for name, result in results.items():
            rows.append({
                'method': result.name,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'accuracy': result.accuracy,
                'avg_confidence': result.avg_confidence
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"✅ Ablation study results saved to {output_path}")

def run_ablation_from_results(results_csv: str, output_csv: str = 'ablation_results.csv'):
    """
    Run ablation study from classification results CSV.
    
    Args:
        results_csv: Path to CSV with columns:
                    - semantic_prediction, semantic_confidence
                    - behavioral_prediction, behavioral_confidence
                    - alpha, true_category
        output_csv: Where to save ablation results
    """
    # Load results
    df = pd.read_csv(results_csv)
    
    # Extract data
    semantic_preds = df['semantic_prediction'].fillna('Others/Uncategorized').tolist()
    semantic_confs = df['semantic_confidence'].fillna(0.0).tolist()
    behavioral_preds = df['behavioral_prediction'].fillna('Others/Uncategorized').tolist()
    behavioral_confs = df['behavioral_confidence'].fillna(0.0).tolist()
    alphas = df['alpha'].fillna(0.5).tolist()
    ground_truth = df['true_category'].tolist()
    
    # Run ablation study
    ablation = AblationBaselines()
    results = ablation.run_ablation_study(
        semantic_preds, semantic_confs,
        behavioral_preds, behavioral_confs,
        alphas, ground_truth
    )
    
    # Print and save results
    ablation.print_comparison_table(results)
    ablation.save_results(results, output_csv)
    
    return results

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        results_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else 'ablation_results.csv'
        run_ablation_from_results(results_csv, output_csv)
    else:
        print("Usage: python ablation_baselines.py <results_csv> [output_csv]")
        print("\nCSV should contain:")
        print("  - semantic_prediction, semantic_confidence")
        print("  - behavioral_prediction, behavioral_confidence")
        print("  - alpha, true_category")

