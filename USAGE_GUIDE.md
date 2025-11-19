# Transaction Categorization System - Usage Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data
```bash
python sample_data_generator.py
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

**OR** on Windows, simply double-click: `run_app.bat`

---

## 📤 Uploading Your Data

### Required CSV Format

Your CSV file must have these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | string | Transaction date & time | `2024-01-15 14:30:00` |
| `amount` | float | Transaction amount | `450.50` |
| `description` | string | Transaction description | `UPI-SWIGGY-REF123456` |
| `type` | string | `credit` or `debit` | `debit` |

**Optional columns:**
- `merchant`: Merchant name (defaults to description if missing)
- `true_category`: Ground truth for validation

### Example CSV:

```csv
date,amount,description,type
2024-01-15 14:30:00,450.50,UPI-SWIGGY-REF123456,debit
2024-01-16 09:00:00,35000.00,Salary Credit - CompanyXYZ,credit
2024-01-16 18:45:00,85.00,Uber Trip Payment,debit
```

---

## 🎯 Using the Application

### Tab 1: Upload & Classify

1. **Upload CSV**: Click "Upload CSV file" and select your transaction file
2. **Review Data**: Check the preview table to ensure data loaded correctly
3. **Start Classification**: Click "🚀 Start Classification"
4. **Wait**: Progress bar shows real-time status
   - Phase 1: Building semantic index (uses first 100 transactions)
   - Phase 2: Extracting behavioral features
   - Phase 3: Clustering transactions (if >20 transactions)
   - Phase 4: Classifying each transaction through all 7 layers

### Tab 2: Results

**Summary Metrics (Top Cards):**
- **Avg Confidence**: Overall confidence across all predictions
- **Auto-Label Rate**: % of transactions classified with confidence ≥75%
- **Categories Found**: Number of unique categories detected
- **Low Confidence**: Count of transactions needing review

**Visualizations:**
- **Category Distribution (Pie Chart)**: Shows transaction breakdown by category
- **Layer Usage (Bar Chart)**: Shows which layer classified each transaction
- **Confidence Distribution (Histogram)**: Distribution of confidence scores

**Detailed Results Table:**
- Filter by category
- Filter by minimum confidence
- See full provenance for each transaction
- Download results as CSV

### Tab 3: Metrics

**Overall Performance:**
- Total transactions processed
- Average confidence score
- Auto-label rate
- Low confidence rate
- Correction rate (if corrections made)

**Layer Performance:**
- Count and percentage for each layer
- Shows which layers are most effective

**Per-Category Metrics** (if ground truth provided):
- Precision, Recall, F1-Score per category
- Support (number of samples)
- Visual comparison chart

**Export:**
- Download metrics as JSON for analysis

### Tab 4: Settings

**Configure thresholds:**
- Unanimous threshold (default: 0.78)
- Majority threshold (default: 0.70)
- Auto-label threshold (default: 0.75)
- Request feedback threshold (default: 0.50)

**Category Management:**
- Enable/disable categories
- View active categories

**System Info:**
- View model configurations
- Check system status

---

## 📊 Understanding Results

### Confidence Levels

| Confidence | Meaning | Action |
|-----------|---------|--------|
| ≥ 0.75 | High confidence | Auto-labeled, no review needed |
| 0.50 - 0.74 | Medium confidence | Probable, low-priority review |
| < 0.50 | Low confidence | Requires user feedback |

### Layer Interpretation

| Layer | What it means |
|-------|---------------|
| **L0: Rule-Based** | Matched deterministic rule (salary, SIP, etc.) |
| **L1: Canonical Match** | Matched known merchant alias |
| **L3: Semantic (gated)** | Text similarity with gating preference |
| **L5: Behavioral (gated)** | Behavioral pattern match with gating |
| **L6: Gated Fusion** | Combined text + behavior |

### Provenance Fields

Each prediction includes:
- **Category**: Predicted category
- **Confidence**: 0.0 - 1.0 score
- **Layer Used**: Which layer made the decision
- **Reason**: Human-readable explanation
- **Alpha (α)**: Gating weight (text vs behavior)

---

## 🎯 Best Practices

### For Best Accuracy

1. **Data Quality**:
   - Ensure dates are properly formatted
   - Remove duplicate transactions
   - Keep descriptions as clean as possible

2. **Data Volume**:
   - Minimum: 50 transactions (basic functionality)
   - Recommended: 200+ transactions (good clustering)
   - Optimal: 500+ transactions (excellent accuracy)

3. **Data Variety**:
   - Include diverse transaction types
   - Cover multiple categories
   - Include both regular and irregular patterns

### Handling Edge Cases

**Vague UPI Transactions** (e.g., "ABCD123@okaxis"):
- System relies on behavioral clustering
- Uses time, amount, recurrence patterns
- Gets anchored to semantic clusters (e.g., ₹45 at 8 AM → Commute)

**New Merchants**:
- First transaction: Uses semantic search or behavior
- Subsequent transactions: Builds behavioral profile
- After 3+ occurrences: Detects recurrence patterns

**Cold Start (New User)**:
- System forces higher text weight (α ≥ 0.7)
- Relies more on semantic matching
- Improves as transaction history grows

---

## 📈 Improving Performance

### If Accuracy is Low

1. **Check Data Quality**:
   - Verify CSV format
   - Ensure correct column names
   - Remove corrupted rows

2. **Add Canonical Aliases**:
   - Edit `layers/layer1_normalization.py`
   - Add your frequently used merchants to `canonical_aliases`

3. **Adjust Thresholds**:
   - Lower auto-label threshold if too conservative
   - Increase if too aggressive

4. **Provide Ground Truth**:
   - Add `true_category` column to CSV
   - System will compute accuracy metrics

### If Processing is Slow

1. **Reduce Data Size**:
   - Start with 100-200 transactions
   - Process in batches

2. **Use GPU** (if available):
   - Edit `requirements.txt`: Change `torch` to `torch+cu118`
   - Change `faiss-cpu` to `faiss-gpu`

3. **Disable Clustering**:
   - Comment out clustering in `app.py` for <20 transactions

---

## 🔧 Advanced Configuration

### Adding New Categories

1. Edit `layers/layer7_classification.py`:
```python
self.categories = [
    'Food & Dining',
    'Your New Category',  # Add here
    ...
]
```

2. Edit `layers/layer1_normalization.py`:
```python
self.category_map = {
    'your_merchant': 'Your New Category',
    ...
}
```

### Adding Custom Rules

Edit `layers/layer0_rules.py`:

```python
def _is_your_custom_rule(self, ...) -> bool:
    # Your logic here
    return True/False
```

Add to `detect()` method:
```python
if self._is_your_custom_rule(...):
    return 'Your Category', 1.0, 'Rule: Your description'
```

### Training Custom Gating Network

1. Collect labeled data (200+ transactions)
2. Train MLP in `layers/layer6_gating.py`
3. Save model weights
4. Load in `GatingController.__init__(model_path='your_model.pt')`

---

## 📊 Metrics Interpretation

### Precision
- **Definition**: Of all predicted category X, how many were correct?
- **High precision**: Few false positives, trustworthy predictions
- **Target**: ≥ 0.85

### Recall
- **Definition**: Of all actual category X, how many did we find?
- **High recall**: Few false negatives, comprehensive coverage
- **Target**: ≥ 0.80

### F1-Score
- **Definition**: Harmonic mean of precision and recall
- **Balanced**: Good overall performance
- **Target**: ≥ 0.82

### Auto-Label Rate
- **Definition**: % of transactions classified with confidence ≥ 0.75
- **High rate**: System is confident in most predictions
- **Target**: 80-92%

### Correction Rate
- **Definition**: % of predictions corrected by user
- **Low rate**: System is accurate
- **Target**: < 5%

---

## 🐛 Troubleshooting

### Error: "Missing required columns"
**Solution**: Ensure CSV has `date`, `amount`, `description`, `type`

### Error: "File content ... exceeds maximum"
**Solution**: Your CSV is too large. Split into smaller files (<10,000 rows)

### Warning: "Empty index"
**Solution**: Not enough data to build semantic index. Need at least 10 transactions.

### Low confidence across all predictions
**Solution**: 
- Add more canonical aliases
- Increase training data
- Provide ground truth for validation

### Application crashes during classification
**Solution**:
- Check for NaN/null values in data
- Ensure dates are valid
- Verify amounts are numeric

---

## 💡 Tips & Tricks

1. **Start Small**: Test with 50-100 transactions first
2. **Iterate**: Review low-confidence predictions and add rules
3. **Feedback Loop**: Correct predictions to improve future accuracy
4. **Export Metrics**: Track performance over time
5. **Use Ground Truth**: Always include `true_category` for validation

---

## 📞 Support

For issues or questions:
1. Check this guide first
2. Review README.md
3. Check individual layer files for implementation details
4. Refer to TRANSACTION_CATEGORIZATION_PLAN.md for system design

---

**Happy Categorizing! 🎉**

