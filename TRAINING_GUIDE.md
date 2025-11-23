# Training Guide for Transaction Categorization System

## Overview

This guide explains how to train all components of the transaction categorization system to match the IEEE conference paper implementation.

## System Architecture

The system consists of 8 layers:

1. **Layer 0**: Rule-Based Detection (corpus-based)
2. **Layer 1**: Text Normalization
3. **Layer 2**: E5 Embeddings (intfloat/e5-base-v2)
4. **Layer 3**: FAISS Semantic Search (with optional attention)
5. **Layer 4**: Behavioral Feature Extraction
6. **Layer 5**: HDBSCAN Clustering
7. **Layer 6**: Gating Network (MLP)
8. **Layer 7**: Final Classification
9. **Layer 8**: Zero-Shot (BART-MNLI) - Optional fallback

## Trainable Components

### 1. Gating Network (Layer 6)

**Purpose**: Learn optimal α (text vs behavior weight) dynamically.

**Training Data Required**:
- `text_confidence`: Confidence from semantic search
- `behavior_confidence`: Confidence from clustering
- `token_count`: Number of tokens in description
- `is_generic_text`: Boolean flag
- `recurrence_confidence`: Recurrence score
- `cluster_density`: Cluster cohesion
- `user_txn_count`: Number of user transactions
- `semantic_consensus`: Agreement among semantic neighbors
- `text_prediction`: Category from semantic layer
- `behavior_prediction`: Category from behavioral layer
- `true_category`: Ground truth label

**Training Command**:
```bash
python layers/gating_trainer.py training_data.csv
```

**Or use the comprehensive training script**:
```bash
python train_system.py training_data.csv --epochs 100
```

**Expected Performance**:
- Validation Loss: < 0.05
- Training Time: ~5-10 minutes (CPU)

### 2. Attention Mechanism (Layer 3 Enhancement)

**Purpose**: Weight different parts of embeddings for better semantic matching.

**Training Data Required**:
- Triplet data: (query, positive, negative) embeddings
- Query: Transaction embedding
- Positive: Similar transaction (same category)
- Negative: Dissimilar transaction (different category)

**Training** (TODO - implement):
```python
from layers.layer3_semantic_search_attention import train_attention_model
train_attention_model('triplet_data.npz', 'models/attention_trained.pt')
```

### 3. Confidence Calibrator

**Purpose**: Calibrate confidence scores to match actual accuracy.

**Training Data Required**:
- `predictions`: Predicted categories
- `confidences`: Raw confidence scores
- `ground_truth`: True categories

**Training**:
```python
from layers.ensemble_classifier import EnsembleClassifier

ensemble = EnsembleClassifier(calibrate=True)
ensemble.calibrate_confidence(predictions, confidences, ground_truth, method='isotonic')
ensemble.save_calibrator('models/confidence_calibrator.pkl')
```

**Or use training script**:
```bash
python train_system.py training_data.csv
```

## Data Preparation

### Step 1: Collect Transaction Data

Required columns:
- `date`: Transaction date (YYYY-MM-DD HH:MM:SS)
- `amount`: Transaction amount (float)
- `description`: Transaction description
- `type`: Transaction type ('credit' or 'debit')
- `merchant`: Merchant name
- `true_category`: Ground truth category (for training)

Optional (recommended for UPI):
- `recipient_name`: UPI recipient name
- `upi_id`: UPI ID
- `note`: Transaction note

### Step 2: Run Initial Classification

```bash
streamlit run app.py
```

Upload your CSV and classify transactions. Download results.

### Step 3: Prepare Training Data

```bash
python train_system.py training_data.csv \
    --prepare-data \
    --transactions transactions.csv \
    --results classification_results.csv
```

This merges transaction data with classification results.

### Step 4: Train All Components

```bash
python train_system.py training_data.csv \
    --output-dir models \
    --epochs 100
```

This trains:
- Gating Network
- Confidence Calibrator
- Runs ablation study

## Evaluation

### Cross-Validation

```python
from evaluation.cross_validation import TransactionCrossValidator

validator = TransactionCrossValidator(n_splits=5, temporal=True)
results = validator.cross_validate(df, classifier_fn)
validator.print_results(results)
validator.save_results(results, 'cv_results.json')
```

### Ablation Study

```python
from evaluation.ablation_baselines import AblationBaselines

ablation = AblationBaselines()
results = ablation.run_ablation_study(
    semantic_preds, semantic_confs,
    behavioral_preds, behavioral_confs,
    alphas, ground_truth
)
ablation.print_comparison_table(results)
```

### Advanced Metrics

```python
from evaluation.advanced_metrics import AdvancedMetricsCalculator

calculator = AdvancedMetricsCalculator(categories)
metrics = calculator.compute_all_metrics(predictions, ground_truth, confidences)
calculator.print_detailed_report(metrics)
calculator.plot_confusion_matrix(predictions, ground_truth)
```

## Expected Performance (IEEE Paper Benchmarks)

### Clustering Quality
- **Silhouette Score**: 0.52 (ours) vs 0.34 (semantic-only), 0.41 (behavioral-only)
- **Davies-Bouldin Index**: 0.72 (ours) vs 1.18 (semantic-only), 0.96 (behavioral-only)
- **V-Measure**: 0.84 (ours) vs 0.65 (semantic-only), 0.70 (behavioral-only)

### Classification Performance
- **Precision**: 0.85+ (weighted)
- **Recall**: 0.80+ (weighted)
- **F1-Score**: 0.82+ (weighted)
- **Auto-Label Rate**: 80-92% (confidence ≥ 0.75)
- **User Correction Rate**: < 5%

### Per-Category Performance
| Category | Precision | Recall | Auto-Label Rate |
|----------|-----------|--------|----------------|
| SIP/Subscription | 0.90+ | 0.85+ | 92% |
| Salary | 0.95+ | 0.90+ | 95% |
| Food & Dining | 0.80-0.88 | 0.75-0.85 | 85% |
| Commute/Transport | 0.85+ | 0.80+ | 88% |
| Bills & Utilities | 0.88+ | 0.85+ | 90% |

## Hyperparameter Tuning

### Gating Network
- **Architecture**: 8 → 128 → 32 → 1
- **Learning Rate**: 0.001
- **Dropout**: 0.2
- **Batch Size**: 32
- **Epochs**: 100
- **Early Stopping**: 15 epochs patience

### Semantic Search
- **Unanimous Threshold**: 0.78
- **Majority Threshold**: 0.70
- **Top-K**: 10

### HDBSCAN Clustering
- **Min Cluster Size**: max(5, 0.01 * n_txns)
- **Min Samples**: 5
- **Cluster Selection Epsilon**: 0.3

### Classification Thresholds
- **Auto-Label**: 0.75
- **Probable**: 0.50
- **Request Feedback**: < 0.50

## Troubleshooting

### Low Gating Network Performance
- **Issue**: Validation loss > 0.10
- **Solution**: 
  - Increase training data (need 500+ samples)
  - Check feature quality
  - Adjust learning rate

### Poor Calibration
- **Issue**: ECE > 0.15
- **Solution**:
  - Use isotonic regression instead of Platt scaling
  - Increase calibration data
  - Check for class imbalance

### Low Clustering Quality
- **Issue**: Silhouette score < 0.40
- **Solution**:
  - Tune HDBSCAN parameters
  - Add more behavioral features
  - Check feature scaling

## Advanced Features

### Enhanced Behavioral Features
Use `EnhancedBehavioralFeatureExtractor` for:
- Velocity features (spending rate changes)
- Sequence patterns (transaction chains)
- Network features (merchant relationships)
- Anomaly detection

```python
from layers.layer4_behavioral_features_enhanced import EnhancedBehavioralFeatureExtractor

extractor = EnhancedBehavioralFeatureExtractor()
features = extractor.extract(txn, user_history)
```

### Attention-Based Semantic Search
```python
from layers.layer3_semantic_search_attention import SemanticSearcherWithAttention

searcher = SemanticSearcherWithAttention(attention_model_path='models/attention.pt')
category, conf, prov = searcher.search(query_embedding)
```

### Ensemble Classification
```python
from layers.ensemble_classifier import EnsembleClassifier

ensemble = EnsembleClassifier(calibrate=True)
category, conf, prov = ensemble.predict_ensemble(predictions, method='weighted_vote')
```

## Production Deployment

### Update app.py to Use Trained Models

```python
# In app.py, update initialization:
gating_controller = GatingController(
    model_path='models/gating_trained.pt',
    use_trained_model=True
)

# Load calibrator
ensemble = EnsembleClassifier()
ensemble.load_calibrator('models/confidence_calibrator.pkl')

# Apply calibration
final_confidence = ensemble.apply_calibration(raw_confidence)
```

### Model Files to Deploy
```
models/
├── gating_trained.pt           # Trained gating network
├── confidence_calibrator.pkl   # Confidence calibrator
└── attention_trained.pt        # (Optional) Attention model
```

## Citation

If you use this system in research, please cite:

```bibtex
@inproceedings{spendwise2024,
  title={SpendWise: Unsupervised Hybrid Transaction Categorization with Adaptive Gating},
  author={Your Name},
  booktitle={IEEE Conference},
  year={2024}
}
```

## Support

For issues or questions:
1. Check this guide
2. Review code comments
3. Run ablation study to diagnose issues
4. Check training logs

## Next Steps

1. ✅ Train gating network
2. ✅ Calibrate confidence scores
3. ✅ Run ablation study
4. ⬜ Train attention mechanism (optional)
5. ⬜ Deploy to production
6. ⬜ Monitor performance
7. ⬜ Collect feedback for retraining

