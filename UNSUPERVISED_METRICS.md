# Unsupervised Metrics for Transaction Categorization

## 🎯 Why Unsupervised Metrics?

This system is **completely unsupervised** - it doesn't require labeled training data or ground truth. Therefore, traditional supervised metrics like:
- ❌ Precision
- ❌ Recall  
- ❌ F1-Score
- ❌ Accuracy
- ❌ Confusion Matrix

**Cannot be computed** (unless user provides optional ground truth for validation).

Instead, we track **operational metrics** that measure system performance without needing ground truth labels.

---

## 📊 Core Unsupervised Metrics

### **1. Confidence Metrics** ⭐⭐⭐
**Most Important** - Measures how certain the system is about its predictions.

| Metric | Description | Target |
|--------|-------------|--------|
| **Avg Confidence** | Mean confidence across all predictions | 0.75+ |
| **Median Confidence** | Middle confidence value | 0.70+ |
| **Std Confidence** | Confidence variability | <0.20 |
| **Confidence Percentiles** | p25, p50, p75, p90, p95 | p75 ≥ 0.80 |

**Why Important**: High confidence = system is certain about predictions.

---

### **2. Auto-Label Rate** ⭐⭐⭐
**Key Metric** - Percentage of transactions that don't need manual review.

| Threshold | Description | Target |
|-----------|-------------|--------|
| **≥ 0.75** | Auto-label (no review needed) | 80-92% |
| **0.50-0.74** | Probable (low-priority review) | 10-15% |
| **< 0.50** | Low confidence (needs review) | <5% |

**Formula**: `Auto-Label Rate = (count with confidence ≥ 0.75) / total transactions`

**Why Important**: Higher auto-label rate = less manual work needed.

---

### **3. Layer Distribution** ⭐⭐
**System Health** - Which layers are classifying transactions.

| Layer | Expected % | Interpretation |
|-------|-----------|----------------|
| **L0: Rules** | 10-20% | Deterministic patterns (salary, SIP) |
| **L1: Canonical** | 30-50% | Known merchant matches |
| **L3: Semantic** | 20-30% | Text similarity |
| **L5: Behavioral** | 10-20% | Pattern-based (vague UPI) |
| **L6: Gated** | 10-20% | Hybrid fusion |

**Why Important**: Shows which layers are effective. If one layer dominates, others may need tuning.

---

### **4. Layer-Wise Confidence** ⭐⭐
**Layer Quality** - Average confidence per layer.

| Layer | Expected Confidence | Why |
|-------|---------------------|-----|
| **L0: Rules** | 0.95+ | Deterministic = high certainty |
| **L1: Canonical** | 0.85+ | Exact match = high certainty |
| **L3: Semantic** | 0.70-0.85 | Similarity-based |
| **L5: Behavioral** | 0.65-0.80 | Pattern-based |

**Why Important**: Identifies which layers are trustworthy.

---

### **5. Category Distribution** ⭐⭐
**Data Balance** - How transactions are distributed across categories.

**Healthy Distribution**:
- No single category > 40%
- No category < 2% (unless genuinely rare)
- Top 3 categories cover 50-70% of transactions

**Warning Signs**:
- 🚨 One category > 70% → May be over-categorizing
- 🚨 Too many in "Others" → Need better rules/aliases

**Why Important**: Shows if categorization is balanced or biased.

---

### **6. Category-Wise Confidence** ⭐⭐
**Category Quality** - Confidence per category.

Expected by category:
| Category | Expected Confidence | Reason |
|----------|---------------------|--------|
| **Salary** | 0.90+ | Strong patterns |
| **Investments** | 0.88+ | SIP patterns clear |
| **Subscriptions** | 0.85+ | Recurring patterns |
| **Food & Dining** | 0.75-0.85 | Common merchants |
| **Transport** | 0.70-0.85 | Mix of clear + vague |
| **Others** | 0.40-0.60 | Catch-all category |

**Why Important**: Shows which categories are well-defined vs ambiguous.

---

### **7. Gating Statistics** ⭐⭐
**Fusion Quality** - How the system balances text vs behavior.

| Metric | Description | Expected |
|--------|-------------|----------|
| **Avg α (Alpha)** | Mean gating weight | 0.40-0.60 |
| **Text Dominant** | % where α ≥ 0.5 | 50-70% |
| **Behavior Dominant** | % where α < 0.5 | 30-50% |

**Interpretation**:
- α = 0.8 → Trusts text (clear merchant)
- α = 0.3 → Trusts behavior (vague UPI)

**Why Important**: Shows if gating is working correctly. If α always high → not using behavioral signals.

---

### **8. Merchant Consistency** ⭐⭐
**Categorization Stability** - Same merchant gets same category.

**Formula**: 
```
For each merchant:
  consistency = (most_common_category_count) / (total_occurrences)

avg_consistency = mean across all merchants with >1 occurrence
```

**Target**: 0.85+ (85%+ consistency)

**Why Important**: Low consistency = merchant getting different categories (bad).

---

### **9. Correction Rate** ⭐⭐⭐
**User Feedback** - How often users correct predictions.

**Formula**: `Correction Rate = (user corrections) / (total predictions)`

**Target**: < 5%

**Why Important**: Direct measure of accuracy from user perspective. Lower = better.

---

### **10. Processing Time** ⭐
**System Performance** - How fast the system processes transactions.

| Metric | Target |
|--------|--------|
| **Avg time per transaction** | < 0.5s |
| **Total processing time** | Depends on volume |

**Why Important**: Shows if system is scalable.

---

### **11. Clustering Quality Metrics** ⭐⭐⭐
**HDBSCAN Performance** - Measures quality of behavioral clustering (as per IEEE paper).

| Metric | Description | Range | Better | Target |
|--------|-------------|-------|--------|--------|
| **Silhouette Score** | Cluster cohesion & separation | -1 to 1 | Higher | 0.52 |
| **Davies-Bouldin Index** | Cluster similarity ratio | 0 to ∞ | Lower | 0.72 |
| **V-Measure** | Homogeneity & completeness | 0 to 1 | Higher | 0.84 |

**Silhouette Score**:
- Measures how similar objects are to their own cluster vs other clusters
- +1: Perfect clustering
- 0: Overlapping clusters  
- -1: Wrong clustering
- **Our approach (adaptive fusion)**: 0.52
- **Baseline (semantic only)**: 0.34

**Davies-Bouldin Index**:
- Measures average similarity of each cluster with its most similar cluster
- 0: Perfect separation
- Higher = worse separation
- **Our approach (adaptive fusion)**: 0.72
- **Baseline (semantic only)**: 1.18

**V-Measure** (requires ground truth):
- Harmonic mean of homogeneity and completeness
- 1.0 = Perfect match with ground truth
- **Our approach (adaptive fusion)**: 0.84
- **Baseline (semantic only)**: 0.65

**Additional Cluster Statistics**:
- `n_clusters`: Number of discovered clusters
- `n_noise_points`: Transactions not assigned to any cluster (HDBSCAN outliers)
- `noise_ratio`: % of noise points
- `avg_cluster_size`: Mean transactions per cluster
- `min/max_cluster_size`: Cluster size range

**Why Important**: Directly validates that the hybrid semantic-behavioral approach is working as described in the IEEE paper.

---

## 📈 Metrics Dashboard Layout

### **Tab 3: Metrics** (in Streamlit App)

```
📊 Overall Performance
├── Total Transactions
├── Unique Categories
├── Avg Confidence
├── Median Confidence
├── Auto-Label Rate (KEY)
├── Probable Rate
├── Low Confidence Rate
└── Correction Rate (KEY)

📈 Confidence Distribution
├── Percentiles (p25, p50, p75, p90, p95)
└── Statistics (min, max, std)

🎯 Layer Performance
├── Layer distribution (count, %)
└── Layer-wise confidence

📊 Category-Wise Performance
├── Count per category
├── Percentage
├── Avg confidence per category
├── Min/Max confidence
└── Auto-label rate per category

⚖️ Gating Network Statistics
├── Avg α (alpha)
├── Text dominant rate
└── Behavior dominant rate

🏪 Merchant Consistency
├── Avg consistency
└── Merchants tracked

⚡ Processing Statistics
├── Avg time per transaction
└── Total processing time

🎯 Clustering Quality Metrics (NEW)
├── Silhouette Score (higher = better)
├── Davies-Bouldin Index (lower = better)
├── V-Measure (if ground truth available)
├── Number of clusters
├── Noise points & ratio
├── Cluster sizes (avg, min, max)
└── Comparison with IEEE Paper Table 1
```

---

## 🎯 Success Criteria (Unsupervised)

Your system is performing well if:

✅ **Auto-label rate**: 80-92%  
✅ **Avg confidence**: 0.75+  
✅ **Correction rate**: <5%  
✅ **Merchant consistency**: 0.85+  
✅ **Layer distribution**: Balanced across layers  
✅ **Category confidence**: Most categories >0.70  
✅ **Silhouette Score**: 0.45+ (target: 0.52)
✅ **Davies-Bouldin Index**: <0.85 (target: 0.72)
✅ **V-Measure** (if ground truth): 0.80+ (target: 0.84)

---

## 📊 Example Metrics Output

```json
{
  "total_transactions": 200,
  "unique_categories": 8,
  "avg_confidence": 0.82,
  "median_confidence": 0.85,
  "std_confidence": 0.15,
  "auto_label_rate": 0.88,
  "probable_rate": 0.09,
  "low_confidence_rate": 0.03,
  "correction_rate": 0.02,
  
  "confidence_percentiles": {
    "p25": 0.72,
    "p50": 0.85,
    "p75": 0.92,
    "p90": 0.96,
    "p95": 0.98
  },
  
  "layer_distribution": {
    "L0: Rules": 15,
    "L1: Canonical Match": 85,
    "L3: Semantic": 60,
    "L5: Behavioral": 30,
    "L6: Gated": 10
  },
  
  "gating_stats": {
    "avg_alpha": 0.55,
    "text_dominant_rate": 0.62,
    "behavior_dominant_rate": 0.38
  },
  
  "merchant_consistency": {
    "avg_consistency": 0.89,
    "merchants_tracked": 45
  }
}
```

---

## 🔍 Interpreting Metrics

### **Scenario 1: Low Auto-Label Rate (e.g., 60%)**

**Problem**: Too many transactions need review.

**Possible Causes**:
- Not enough canonical aliases → Add more merchants
- Weak behavioral signals → Need more training data
- Gating too conservative → Tune thresholds

**Fix**:
1. Check layer distribution - which layer is failing?
2. Add canonical aliases for common merchants
3. Lower auto-label threshold from 0.75 to 0.70 (if confident)

---

### **Scenario 2: Low Merchant Consistency (e.g., 65%)**

**Problem**: Same merchant getting different categories.

**Possible Causes**:
- Merchant has multiple business types (Amazon = shopping + subscriptions)
- Noisy text normalization
- Weak clustering

**Fix**:
1. Check which merchants have low consistency
2. Add merchant-specific rules
3. Improve text normalization
4. Use transaction amount as secondary signal

---

### **Scenario 3: High Correction Rate (e.g., 15%)**

**Problem**: Users frequently disagree with predictions.

**Possible Causes**:
- Wrong canonical mappings
- Bad rules (e.g., salary threshold too low)
- Behavioral patterns not learned

**Fix**:
1. Analyze correction history - which categories are corrected most?
2. Update rules based on corrections
3. Add corrected examples to semantic index
4. Retrain gating network

---

### **Scenario 4: One Layer Dominates (e.g., L1 = 90%)**

**Problem**: Other layers not being used.

**If L1 dominates**:
- ✅ Good! Means most merchants are known
- Consider: Is semantic search being bypassed?

**If L0 dominates**:
- 🚨 May be over-relying on rules
- Other layers may have low confidence thresholds

**If L5 dominates**:
- 🚨 Too many vague merchants
- Need better text normalization and canonical aliases

**Fix**:
- Balance thresholds across layers
- Ensure each layer has reasonable confidence requirements

---

## 🎓 Key Insights

### **1. Auto-Label Rate = Primary Success Metric**
In unsupervised systems, the goal is to **minimize manual review**. Auto-label rate directly measures this.

### **2. Correction Rate = Ground Truth Proxy**
User corrections are the **only source of truth** in unsupervised systems. Track corrections to measure real-world accuracy.

### **3. Confidence Calibration is Critical**
System confidence should match real-world accuracy:
- High confidence → Should be correct
- Low confidence → Should prompt for review

### **4. Layer Distribution Shows System Balance**
Healthy distribution means **all layers contribute**. If one layer dominates, system may be under-utilizing other signals.

### **5. Merchant Consistency = Long-Term Quality**
Same merchant should consistently get same category. Consistency improves over time with corrections and feedback.

---

## 📝 Metrics Export Format

Metrics can be exported to JSON:

```json
{
  "timestamp": "2024-11-18T10:30:00",
  "total_transactions": 200,
  "avg_confidence": 0.82,
  "auto_label_rate": 0.88,
  "correction_rate": 0.02,
  "layer_distribution": {...},
  "category_distribution": {...},
  "gating_stats": {...},
  "merchant_consistency": {...}
}
```

Use exported metrics for:
- Tracking performance over time
- A/B testing different thresholds
- Reporting to stakeholders

---

## 🚀 Monitoring in Production

### **Daily Monitoring**
- ⚠️ Auto-label rate drops below 70%
- ⚠️ Correction rate above 10%
- ⚠️ Avg confidence drops below 0.65

### **Weekly Review**
- 📊 Layer distribution changes
- 📊 Category confidence trends
- 📊 Merchant consistency over time

### **Monthly Analysis**
- 📈 Correction patterns
- 📈 New merchants added
- 📈 Threshold adjustments needed

---

## ✅ Summary

**Unsupervised metrics focus on**:
1. **Confidence** → How certain is the system?
2. **Auto-label rate** → How much manual work is needed?
3. **Correction rate** → How often are we wrong?
4. **Consistency** → Are predictions stable?
5. **Performance** → Is it fast enough?

**No ground truth needed!** These metrics measure operational performance in production without requiring labeled data.

---

**Last Updated**: November 18, 2024  
**Status**: ✅ Fully Implemented in `metrics/metrics_tracker.py`

