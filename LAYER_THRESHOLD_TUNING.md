# Layer Threshold Tuning Guide

## Overview
This guide documents all threshold adjustments made to achieve proper layer distribution and prevent early layers from being too aggressive.

---

## 🎯 Target Distribution

**Ideal layer usage percentages:**
- **Layer 0 (Rules)**: 10-15% (only top brands and obvious patterns)
- **Layer 1 (Canonical)**: 5-8% (only exact matches to top brands)
- **Layer 3 (Semantic)**: 40-50% (majority of merchant matching)
- **Layer 5 (Clustering)**: 20-30% (behavioral patterns)
- **Layer 7 (Final Fusion)**: 10-15% (uncertain cases)
- **Layer 8 (Zero-Shot)**: <5% (extreme fallback)

---

## 📊 Layer 0: Rule-Based Detection

### Location
`layers/layer0_rules.py`

### Changes Made

#### 1. Corpus Matching - Ultra Restrictive
```python
# Strategy 4: ONLY top national brands (very restrictive)
top_national_brands = [
    'netflix', 'amazon', 'flipkart', 'swiggy', 'zomato', 
    'spotify', 'hotstar', 'prime', 'uber', 'ola'
]
if best_keyword in top_national_brands and best_ratio > 0.3:  # Increased from 0.2
    confidence = 0.85
    return best_category, confidence, f'Top brand match: "{best_keyword}"'

# EVERYTHING ELSE goes to Layer 3/5
return None
```

**Key Points:**
- Reduced brand list from ~20 to just 10 top national brands
- Increased ratio threshold from 0.2 to 0.3
- Layer 0 should only catch 10-15% of transactions (not 30%+)

#### 2. Subscription Detection - Dramatically Reduced

**Known Subscription List:**
```python
self.known_subscriptions = {
    # Only top 5 streaming
    'netflix', 'netflixupi', 'amazon prime', 'hotstar', 'spotify',
    # Only top 3 software/cloud
    'youtube premium', 'microsoft 365', 'adobe',
    # Top 2 fitness
    'cult.fit', 'cultfit'
}
```

**Previous list had 25+ services** → Now only **10 top services**

**Multi-Keyword Requirement:**
```python
# If NOT a top-10 service, require MULTIPLE explicit keywords
explicit_keywords = ['subscription', 'membership', 'premium plan', 'renewal', 'recurring payment']
keyword_count = sum(1 for kw in explicit_keywords if kw in combined_text or kw in note)

if keyword_count < 2:  # Need at least 2 explicit keywords
    return False  # Let semantic/clustering layers handle it
```

**Key Points:**
- Reduced known subscription services from 25+ to 10
- Non-top-10 services need **at least 2 explicit keywords** to be classified as subscriptions
- Amount range check (₹50-₹3000) still applies
- Recurrence pattern check still applies

---

## 📊 Layer 1: Normalization & Canonical Matching

### Location
`app.py` (lines 278-312) and `layers/layer1_normalization.py`

### Changes Made

#### 1. Canonical Match Threshold - Much Stricter
```python
# DISABLED: Layer 1 canonical match - too aggressive
# Only keep this for VERY high confidence (>0.98) AND top brands only
if norm_metadata.get('canonical_match') and norm_metadata.get('canonical_confidence', 0) > 0.98:
    canonical = norm_metadata['canonical_match']
    top_brands = ['netflix', 'amazon', 'swiggy', 'zomato', 'uber', 'ola', 'spotify']
    if canonical in top_brands:
        # Classify with Layer 1
        ...
```

**Key Changes:**
- Threshold increased from **0.90 → 0.98** (near-perfect match required)
- Must also be in the `top_brands` list (only 7 services)
- Previously caught 20-30% of transactions → Now catches <5%

#### 2. Recipient Name Matching - Stricter
```python
# layers/layer1_normalization.py
# Enhanced: Check recipient name directly - ONLY if very high confidence
if not canonical and recipient and len(recipient) > 5:
    recipient_canonical, recipient_conf = self._match_canonical(recipient)
    if recipient_canonical and recipient_conf > 0.95:  # Much stricter
        canonical = recipient_canonical
        confidence = recipient_conf
```

**Key Changes:**
- Added minimum length check (>5 chars)
- Raised confidence threshold to **0.95** (was any confidence)
- Removed note field check (too many false positives)

---

## 📊 Layer 3: Semantic Search

### Location
`layers/layer3_semantic_search.py`

### Current Thresholds
```python
# Unanimous top-3 agreement
if unanimous_top3 and max_similarity >= 0.78:
    consensus = "unanimous_top3"
    return category, confidence, consensus

# Majority top-10 agreement
elif majority_top10 and max_similarity >= 0.70:
    consensus = "majority_top10"
    return category, confidence, consensus

# No strong consensus
else:
    return None  # Let other layers handle it
```

**These thresholds are GOOD** - no changes needed here. Layer 3 should be the workhorse.

---

## 📊 Layer 5: Clustering

### Location
`layers/layer5_clustering.py`

### HDBSCAN Parameters (Dynamic)
```python
if n_samples < 50:
    min_cluster_size = 3
    min_samples = 2
elif n_samples < 100:
    min_cluster_size = 5
    min_samples = 3
elif n_samples < 500:
    min_cluster_size = max(3, int(0.03 * n_samples))  # 3% of data
    min_samples = max(2, int(0.01 * n_samples))       # 1% of data
else:
    min_cluster_size = max(5, int(0.02 * n_samples))  # 2% of data
    min_samples = max(3, int(0.008 * n_samples))      # 0.8% of data

self.clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric='euclidean',
    cluster_selection_epsilon=0.1,  # Reduced from 0.3
    cluster_selection_method='eom',  # Excess of Mass
    prediction_data=True
)
```

**Key Points:**
- Dynamic parameters based on dataset size
- More granular clusters (epsilon=0.1 instead of 0.3)
- Should produce 5-15 clusters (not just 2-3)

---

## 📊 Layer 7: Final Classification

### Location
`layers/layer7_classification.py`

### Auto-Label Threshold
```python
# Lowered from 0.75 to 0.70 to reduce Layer 8 usage
if confidence >= 0.70:
    auto_label = True
```

**Key Point:** Slightly more lenient to avoid triggering Layer 8 unnecessarily.

---

## 📊 Layer 8: Zero-Shot (Fallback)

### Location
`app.py`

### Usage Restrictions
```python
# Layer 8: Zero-shot (LAST RESORT)
use_zeroshot = (
    st.session_state.get('use_zeroshot', False) and  # Must be enabled
    not rule_result['detected'] and                   # L0 failed
    (semantic_category is None or semantic_confidence < 0.40) and  # L3 failed
    cluster_confidence < 0.40                         # L5 failed
)
```

**Key Points:**
- Must be explicitly enabled by user
- Only triggers if ALL previous layers have low confidence (<0.40)
- Should handle <5% of transactions

---

## 🎯 Testing & Validation

### How to Check Distribution

After classifying transactions, check the "📊 Layer Usage" tab:

**Expected Distribution:**
```
Layer 0 (Rules):          10-15%
Layer 1 (Canonical):      5-8%
Layer 3 (Semantic):       40-50%  ← MAJORITY
Layer 5 (Clustering):     20-30%
Layer 7 (Final Fusion):   10-15%
Layer 8 (Zero-Shot):      <5%
```

### Red Flags

⚠️ **Layer 0 > 25%** → Corpus matching too aggressive, raise thresholds  
⚠️ **Layer 1 > 15%** → Canonical matching too aggressive, raise confidence threshold  
⚠️ **Layer 3 < 30%** → Early layers blocking too much, reduce L0/L1 usage  
⚠️ **Layer 8 > 10%** → Thresholds too strict across all layers, need adjustment  

---

## 🔧 Fine-Tuning Recommendations

### If Layer 0 is still too high (>20%):

1. **Reduce the brand list further** in `layers/layer0_rules.py`:
   ```python
   top_national_brands = [
       'netflix', 'amazon', 'swiggy', 'zomato', 'uber'  # Only 5 brands
   ]
   ```

2. **Increase ratio threshold**:
   ```python
   if best_keyword in top_national_brands and best_ratio > 0.4:  # Was 0.3
   ```

### If Layer 3 is too low (<35%):

1. **Lower semantic thresholds** in `layers/layer3_semantic_search.py`:
   ```python
   if unanimous_top3 and max_similarity >= 0.75:  # Was 0.78
   ```

2. **Further restrict Layer 1**:
   ```python
   # Disable canonical matching entirely
   # Just comment out the entire canonical match section in app.py
   ```

### If subscriptions are still over-classified:

1. **Reduce the known subscription list further**:
   ```python
   self.known_subscriptions = {
       'netflix', 'netflixupi', 'amazon prime', 'spotify'  # Only 4 top services
   }
   ```

2. **Require recurrence + keywords**:
   - Add logic to require BOTH recurring pattern AND explicit keywords for non-top-4 services

---

## 📝 Summary of Changes

| Layer | Previous Behavior | New Behavior | Expected % |
|-------|-------------------|--------------|-----------|
| **L0** | Matched 20+ brands, ratio >0.2 | Matches 10 brands, ratio >0.3 | 10-15% |
| **L0 Subs** | 25+ services, 1 keyword | 10 services, 2 keywords | Reduced 70% |
| **L1** | Confidence >0.90, any brand | Confidence >0.98, 7 brands only | 5-8% |
| **L3** | Thresholds good | No changes | 40-50% |
| **L5** | Fixed params, 2-3 clusters | Dynamic params, 5-15 clusters | 20-30% |
| **L7** | Threshold 0.75 | Threshold 0.70 | 10-15% |
| **L8** | Triggered if <0.50 | Triggered if <0.40 | <5% |

---

## 🚀 Next Steps

1. **Test with real data** and check layer distribution
2. **Adjust thresholds** based on results (see fine-tuning section above)
3. **Monitor subscription over-classification** in the results tab
4. **Check cluster quality** in the "🔍 Clusters" tab
5. **Iterate** until distribution matches the target

---

Last Updated: 2025-11-19  
Version: 2.0 (Major threshold overhaul)

