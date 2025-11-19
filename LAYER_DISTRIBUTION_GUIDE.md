# Layer Distribution Guide

## 🎯 Goal: Normal Distribution Across Layers

You want transactions to be **distributed normally** across layers, not concentrated in Layer 0. Here's how the system is now configured:

---

## 📊 Target Layer Distribution

### **Ideal Distribution (Normal/Balanced):**

```
Layer 0 (Rules):           20-30%  ← Only COMMON, OBVIOUS merchants
Layer 1 (Canonical):       5-10%   ← Known merchant aliases  
Layer 3 (Semantic):        30-40%  ← Primary semantic matching
Layer 5 (Behavioral):      20-30%  ← Pattern-based clustering
Layer 8 (Zero-shot):       <5%     ← Difficult cases only
Others/Uncategorized:      <5%     ← Need more data
```

### **What Changed:**

**Before (Over-controlled by Layer 0):**
```
Layer 0: 70-80%  ❌ Too dominant
Layer 3: 10-15%  ❌ Underutilized
Layer 5: 5-10%   ❌ Underutilized
```

**After (Balanced):**
```
Layer 0: 20-30%  ✅ Common merchants only
Layer 3: 30-40%  ✅ Primary classification
Layer 5: 20-30%  ✅ Pattern detection
```

---

## 🔧 Changes Made

### **1. Layer 0: More Selective (Only Common Merchants)**

**New Strategy - 4 Tiers:**

#### **Tier 1: Exact Match (98% confidence)**
```python
Example: "netflix" exactly matches "netflix"
→ Returns immediately
```

#### **Tier 2: Dominant Match (95% confidence)**
```python
Example: "netflix subscription" 
→ "netflix" is >50% of text AND keyword length ≥5
→ Returns
```

#### **Tier 3: High Quality Partial (90% confidence)**
```python
Example: "subscription netflix premium"
→ Match ratio >30% AND keyword length >8 characters
→ Returns
```

#### **Tier 4: Common Brand Keywords (85% confidence)**
```python
Common brands list:
  netflix, amazon, flipkart, swiggy, zomato, uber, ola,
  spotify, hotstar, prime, paytm, phonepe, googlepay,
  starbucks, mcdonalds, kfc, dominos, indigo, air india

Example: "payment to amazon"
→ Match ratio >20% AND keyword in common brands
→ Returns
```

#### **Tier 5: Everything Else → Let Other Layers Handle**
```python
Example: "vinayak", "julfikar bakery", "local merchant"
→ NOT in common brands
→ Returns None (Layer 3 or 5 will handle)
```

---

### **2. Clustering: More Granular (More Clusters)**

**Improved HDBSCAN Parameters:**

| Dataset Size | Old min_cluster_size | New min_cluster_size | Effect |
|--------------|---------------------|---------------------|--------|
| <50 txns | 5 | **3** | More sensitive |
| 50-100 | 5 | **5** | More clusters |
| 100-500 | ~1% of data | **3% of data** | 3x more clusters |
| >500 | ~1% of data | **2% of data** | 2x more clusters |

**Other Improvements:**
- `cluster_selection_epsilon`: 0.3 → **0.1** (more clusters)
- `cluster_selection_method`: 'leaf' → **'eom'** (better for varied densities)
- Dynamic `min_samples` based on dataset size

**Expected Result:**
- **Before:** 3-5 clusters
- **After:** 8-15 clusters (for 100+ transactions)

---

## 📈 How Layers Work Together Now

### **Transaction Flow:**

```
┌─────────────────────────────────────────────────────┐
│ TRANSACTION: "Payment to Vinayak - UPI"            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │ Layer 0: Rules              │
        │ Check: Common merchants?    │
        └─────────────┬───────────────┘
                      │
                      ├─ Is it Netflix/Amazon/etc? → YES → Classify (20-30%)
                      │
                      └─ Is it Vinayak/local name? → NO → Pass to Layer 3
                                    │
                                    ▼
                      ┌─────────────────────────────┐
                      │ Layer 3: Semantic Search    │
                      │ Find similar transactions   │
                      └─────────────┬───────────────┘
                                    │
                                    ├─ High similarity? → Classify (30-40%)
                                    │
                                    └─ No consensus? → Pass to Layer 5
                                                  │
                                                  ▼
                                    ┌─────────────────────────────┐
                                    │ Layer 5: Clustering         │
                                    │ Check behavioral pattern    │
                                    └─────────────┬───────────────┘
                                                  │
                                                  ├─ Fits cluster? → Classify (20-30%)
                                                  │
                                                  └─ No pattern? → Layer 8 or Others (<5%)
```

---

## 🎯 Examples

### **Example 1: Netflix (Layer 0)**
```
Transaction: "NETFLIX netflixupi Monthly"
→ Layer 0: "netflix" is exact match in common brands
→ Result: Layer 0 (98% confidence) ✅
→ Category: Subscriptions
```

### **Example 2: JULFIKAR Bakery (Layer 3)**
```
Transaction: "JULFIKAR paytmqr1jc baker"
→ Layer 0: "julfikar" NOT in common brands → Pass
→ Layer 3: Search index, find similar "baker" transactions
→ Result: Layer 3 (85% confidence) ✅
→ Category: Food & Dining
```

### **Example 3: VINAYAK Transfer (Layer 5)**
```
Transaction: "VINAYAK vinayakpbh UPI" ₹943
→ Layer 0: "vinayak" NOT common brand → Pass
→ Layer 3: No strong semantic match → Pass
→ Layer 5: Similar amount (₹943), similar recipients (ANUSHKA, SHUBHAM)
→ Result: Layer 5 (80% confidence) ✅
→ Category: Transfers
```

### **Example 4: New Local Restaurant (Layer 3/5)**
```
Transaction: "SNOWCRE paytmqr281 UPI"
→ Layer 0: "snowcre" NOT common brand → Pass
→ Layer 3: Sequential learning - check if seen before
  - If similar transactions exist → Layer 3
  - If not → Layer 5 (pattern based)
→ Result: Layer 3 or 5 (70-80% confidence)
→ Category: Food & Dining (learned over time)
```

---

## 🔍 Expected Cluster Distribution

With improved parameters, you should now see:

### **For 25 Transactions:**
```
Expected: 5-8 clusters

Cluster -1 (Noise): 20-30% (5-7 txns)
  - One-time purchases
  - Unique patterns

Cluster 0: 15-20% (4-5 txns)
  - Example: Transfers (VINAYAK, ANUSHKA, SHUBHAM)

Cluster 1: 10-15% (2-4 txns)
  - Example: Food delivery (JULFIKAR, BIKANER)

Cluster 2: 10-15% (2-4 txns)
  - Example: Subscriptions (NETFLIX)

Cluster 3: 10-15% (2-4 txns)
  - Example: Transport (IndianR)

... more clusters as patterns emerge
```

### **For 100+ Transactions:**
```
Expected: 10-15 clusters

- Daily coffee (high frequency, small amounts)
- Subscriptions (monthly, fixed amounts)
- Person transfers (similar amounts, different people)
- Transport (commute times, fuel stations)
- Groceries (weekly, moderate amounts)
- Entertainment (weekends, variable amounts)
- Bills (monthly, month start)
- Shopping (irregular, varying amounts)
- Healthcare (infrequent, larger amounts)
- Education (semester patterns)
... and more
```

---

## ⚙️ Fine-Tuning (If Needed)

### **If Layer 0 Still Too Dominant (>40%):**

**Option 1:** Remove less common merchants from corpus
```python
# Edit data/mumbai_merchants_corpus.json
# Keep only: Netflix, Amazon, Swiggy, Zomato, Uber, etc.
# Remove: Local names, less common merchants
```

**Option 2:** Increase match ratio threshold
```python
# In layers/layer0_rules.py, line 146-147
if best_ratio > 0.5:  # Change from 0.3 to 0.5 (stricter)
```

### **If Too Few Clusters (<5):**

**Option 1:** Reduce min_cluster_size further
```python
# In layers/layer5_clustering.py, line 62
min_cluster_size = max(2, int(0.02 * n_samples))  # Change 0.03 to 0.02
```

**Option 2:** Reduce cluster_selection_epsilon
```python
# In layers/layer5_clustering.py, line 90
cluster_selection_epsilon=0.05,  # Change from 0.1 to 0.05
```

### **If Too Many Clusters (>20):**

**Option 1:** Increase min_cluster_size
```python
min_cluster_size = max(5, int(0.05 * n_samples))  # Change 0.03 to 0.05
```

**Option 2:** Increase cluster_selection_epsilon
```python
cluster_selection_epsilon=0.2,  # Change from 0.1 to 0.2
```

---

## 📊 Monitoring Layer Distribution

### **Check in Metrics Tab:**

Look for "Layer Distribution" chart:

```
✅ Good (Balanced):
  L0: 25%
  L3: 35%
  L5: 25%
  L8: 5%

⚠️ Over-controlled:
  L0: 70%
  L3: 15%
  L5: 10%

⚠️ Under-performing:
  L0: 5%
  L3: 30%
  L8: 40%  ← Too much AI usage
```

### **Check in Clusters Tab:**

Look for:
- **Number of clusters**: Should be 5-15 for 100+ transactions
- **Noise ratio**: Should be 15-25% (not too high)
- **Silhouette Score**: >0.4 is good
- **Cluster sizes**: Should be varied (not all equal or all different)

---

## 🎉 Summary

**Changes Made:**

1. ✅ **Layer 0 More Selective**
   - Only common, obvious merchants (Netflix, Amazon, etc.)
   - 4-tier strategy with strict thresholds
   - Local/personal names passed to other layers

2. ✅ **Better Clustering**
   - Smaller min_cluster_size (more granular)
   - Dynamic parameters based on data size
   - Better algorithm settings (EOM method)

3. ✅ **Expected Results**
   - Layer 0: 20-30% (not 70-80%)
   - Layer 3: 30-40% (primary)
   - Layer 5: 20-30% (patterns)
   - 8-15 clusters (not 3)

**Test and adjust using the fine-tuning options above!** 🎯

