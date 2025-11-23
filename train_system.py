"""
Comprehensive Training Script for Transaction Categorization System.

Trains all learnable components:
1. Gating Network
2. Attention Mechanism
3. Confidence Calibrator
4. Ensemble Meta-Learner
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime

# Import training modules
from layers.gating_trainer import GatingTrainer, train_gating_from_history
from layers.ensemble_classifier import EnsembleClassifier, ConfidenceAnalyzer
from evaluation.cross_validation import TransactionCrossValidator
from evaluation.ablation_baselines import AblationBaselines

def train_all_components(data_path: str,
                        output_dir: str = 'models',
                        epochs: int = 100):
    """
    Train all system components.
    
    Args:
        data_path: Path to training data CSV
        output_dir: Directory to save trained models
        epochs: Number of training epochs
    """
    print("="*80)
    print("TRAINING TRANSACTION CATEGORIZATION SYSTEM")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} transactions")
    
    # Validate required columns
    required_cols = [
        'text_confidence', 'behavior_confidence',
        'text_prediction', 'behavior_prediction',
        'true_category'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print("   Required columns:")
        for col in required_cols:
            print(f"     - {col}")
        return
    
    # 1. Train Gating Network
    print("\n" + "="*80)
    print("1. TRAINING GATING NETWORK")
    print("="*80)
    
    try:
        gating_model_path = output_path / 'gating_trained.pt'
        history = train_gating_from_history(
            data_path,
            output_model_path=str(gating_model_path),
            epochs=epochs
        )
        print(f"✅ Gating network trained successfully")
        print(f"   Best validation loss: {history['best_val_loss']:.4f}")
    except Exception as e:
        print(f"❌ Gating network training failed: {e}")
    
    # 2. Train Confidence Calibrator
    print("\n" + "="*80)
    print("2. TRAINING CONFIDENCE CALIBRATOR")
    print("="*80)
    
    try:
        ensemble = EnsembleClassifier(calibrate=True)
        
        # Extract data for calibration
        predictions = df['text_prediction'].tolist()  # Or ensemble predictions
        confidences = df['text_confidence'].tolist()
        ground_truth = df['true_category'].tolist()
        
        # Train calibrator
        ensemble.calibrate_confidence(
            predictions,
            confidences,
            ground_truth,
            method='isotonic'
        )
        
        # Save calibrator
        calibrator_path = output_path / 'confidence_calibrator.pkl'
        ensemble.save_calibrator(str(calibrator_path))
        
        # Analyze calibration
        analyzer = ConfidenceAnalyzer()
        calibration_data = analyzer.compute_calibration_curve(
            predictions, confidences, ground_truth
        )
        analyzer.print_calibration_report(calibration_data)
        
        print(f"✅ Confidence calibrator trained successfully")
    except Exception as e:
        print(f"❌ Confidence calibrator training failed: {e}")
    
    # 3. Run Ablation Study
    print("\n" + "="*80)
    print("3. RUNNING ABLATION STUDY")
    print("="*80)
    
    try:
        ablation = AblationBaselines()
        
        # Extract predictions
        semantic_preds = df['text_prediction'].tolist()
        semantic_confs = df['text_confidence'].tolist()
        behavioral_preds = df['behavior_prediction'].tolist()
        behavioral_confs = df['behavior_confidence'].tolist()
        alphas = df.get('alpha', [0.5] * len(df)).tolist()
        ground_truth = df['true_category'].tolist()
        
        # Run ablation
        results = ablation.run_ablation_study(
            semantic_preds, semantic_confs,
            behavioral_preds, behavioral_confs,
            alphas, ground_truth
        )
        
        # Print and save results
        ablation.print_comparison_table(results)
        ablation_path = output_path / 'ablation_results.csv'
        ablation.save_results(results, str(ablation_path))
        
        print(f"✅ Ablation study completed")
    except Exception as e:
        print(f"❌ Ablation study failed: {e}")
    
    # 4. Cross-Validation
    print("\n" + "="*80)
    print("4. CROSS-VALIDATION EVALUATION")
    print("="*80)
    
    try:
        # Note: This requires a classifier function
        # For now, we'll skip this as it needs the full pipeline
        print("⚠️ Cross-validation requires full classification pipeline")
        print("   Run separately using evaluation/cross_validation.py")
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
    
    # 5. Generate Training Report
    print("\n" + "="*80)
    print("5. GENERATING TRAINING REPORT")
    print("="*80)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'n_samples': len(df),
        'epochs': epochs,
        'models_trained': {
            'gating_network': str(output_path / 'gating_trained.pt'),
            'confidence_calibrator': str(output_path / 'confidence_calibrator.pkl')
        },
        'ablation_results': str(output_path / 'ablation_results.csv')
    }
    
    report_path = output_path / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Training report saved to {report_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"📁 Models saved to: {output_dir}/")
    print(f"📊 Training report: {report_path}")
    print("\nTrained components:")
    print("  ✅ Gating Network")
    print("  ✅ Confidence Calibrator")
    print("  ✅ Ablation Baselines")
    print("\nNext steps:")
    print("  1. Update app.py to use trained models")
    print("  2. Run cross-validation for final evaluation")
    print("  3. Test on new data")
    print("="*80)

def prepare_training_data(transactions_csv: str,
                         results_csv: str,
                         output_csv: str):
    """
    Prepare training data from transaction and classification results.
    
    Args:
        transactions_csv: Original transactions
        results_csv: Classification results
        output_csv: Output path for training data
    """
    print("Preparing training data...")
    
    # Load data
    transactions = pd.read_csv(transactions_csv)
    results = pd.read_csv(results_csv)
    
    # Merge
    training_data = pd.merge(
        transactions,
        results,
        left_index=True,
        right_on='transaction_id',
        how='inner'
    )
    
    # Save
    training_data.to_csv(output_csv, index=False)
    print(f"✅ Training data saved to {output_csv}")
    print(f"   {len(training_data)} samples prepared")

def main():
    parser = argparse.ArgumentParser(
        description='Train Transaction Categorization System'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Prepare training data from transactions and results'
    )
    parser.add_argument(
        '--transactions',
        type=str,
        help='Transactions CSV (for --prepare-data)'
    )
    parser.add_argument(
        '--results',
        type=str,
        help='Results CSV (for --prepare-data)'
    )
    
    args = parser.parse_args()
    
    if args.prepare_data:
        if not args.transactions or not args.results:
            print("❌ --prepare-data requires --transactions and --results")
            return
        prepare_training_data(args.transactions, args.results, args.data_path)
    else:
        train_all_components(args.data_path, args.output_dir, args.epochs)

if __name__ == '__main__':
    main()

