# Quick Reference Guide

## 🚀 Common Commands

### Training
```bash
# Train all components
python train_system.py training_data.csv --epochs 100

# Train gating network only
python layers/gating_trainer.py training_data.csv

# Prepare training data
python train_system.py training_data.csv \
    --prepare-data \
    --transactions transactions.csv \
    --results results.csv
```

### Evaluation
```bash
# Run ablation study
python evaluation/ablation_baselines.py results.csv

# Cross-validation (in Python)
python -c "from evaluation.cross_validation import TransactionCrossValidator; ..."
```

### Running the App
```bash
# Standard mode
streamlit run app.py

# With trained models
streamlit run app.py  # Models auto-loaded from models/
```

---

## 📊 Key Metrics to Track

### Clustering Quality
- **Silhouette Score**: Target > 0.50
- **Davies-Bouldin Index**: Target < 0.80
- **V-Measure**: Target > 0.80

### Classification Performance
- **F1-Score**: Target > 0.82
- **Auto-Label Rate**: Target > 80%
- **ECE (Calibration)**: Target < 0.10

---

## 🔧 Configuration Files

### Model Paths
```
models/
├── gating_trained.pt           # Gating network weights
├── confidence_calibrator.pkl   # Confidence calibrator
└── attention_trained.pt        # (Optional) Attention weights
```

### Data Format
```csv
date,amount,description,type,merchant,true_category
2024-01-01 10:30:00,500.00,Swiggy Order,debit,Swiggy,Food & Dining
```

---

## 🐛 Troubleshooting

### Issue: Low gating network performance
**Solution**: Increase training data (need 500+ samples)

### Issue: Poor calibration (ECE > 0.15)
**Solution**: Use isotonic regression, increase calibration data

### Issue: Low clustering quality
**Solution**: Tune HDBSCAN parameters, add more features

---

## 📞 Quick Help

- **Training Guide**: See `TRAINING_GUIDE.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Code Documentation**: Check inline comments
- **System Overview**: See `README.md`

