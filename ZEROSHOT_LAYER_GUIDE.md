# Layer 8: Zero-Shot Classification (BART-MNLI)

## 🎯 Purpose

**Layer 8** is an **optional fallback layer** that uses **BART-MNLI** (Natural Language Inference) to classify transactions when all other methods fail.

---

## 🤖 What is BART-MNLI?

**BART** (Bidirectional and Auto-Regressive Transformers) is a large language model fine-tuned on **MNLI** (Multi-Genre Natural Language Inference) dataset.

**Key Features**:
- **Zero-shot classification**: No training data needed
- **Natural Language Inference**: Tests if premise entails hypothesis
- **Robust**: Works on any text, even very vague descriptions
- **Pre-trained**: facebook/bart-large-mnli (~1.5GB)

---

## 📊 When is Layer 8 Used?

**Priority Order**:
1. ✅ L0: Rule-based → If matched, done
2. ✅ L1: Canonical aliases → If matched, done  
3. ✅ L3: Semantic search → If consensus, continue
4. ✅ L5: Behavioral clustering → If cluster found, continue
5. ✅ L6: Gating → Fuse semantic + behavioral
6. ⚠️ **L8: Zero-shot** → **Only if semantic AND behavioral both fail**
7. ❌ "Others/Uncategorized" → If zero-shot also fails

**Trigger Condition**:
```python
if not semantic_result[0] and not behavioral_result[0]:
    # Use zero-shot as last resort
    zeroshot_result = zeroshot_classifier.classify(...)
```

---

## 🎯 How It Works

### **Method 1: Standard Zero-Shot**

```python
# Input
premise = "UPI-ABCD123@okaxis ₹45 debit"

# Process
For each category in [Food, Transport, Shopping, ...]:
    Score = model.predict_entailment(
        premise, 
        "This transaction is about {category}"
    )

# Output
Category with highest score
```

### **Method 2: NLI Approach** (Alternative)

```python
# Input
premise = "UPI-ABCD123@okaxis ₹45 debit"

# Custom hypotheses
hypotheses = {
    'Food & Dining': 'This transaction is for food, dining, or groceries',
    'Transport': 'This transaction is for transportation or commute',
    ...
}

# Process
For each category, hypothesis:
    Entailment_score = model(premise, hypothesis)

# Output
Category with highest entailment
```

---

## 📊 Performance Characteristics

| Metric | Value |
|--------|-------|
| **Accuracy** | 70-80% (on vague transactions) |
| **Speed** | Slow (~2-3x regular layers) |
| **Model Size** | 1.5 GB |
| **Confidence Threshold** | ≥0.60 to accept |
| **Confidence Discount** | ×0.85 (to prefer other layers) |

---

## ⚙️ Configuration

### **Enable Zero-Shot** (in Streamlit UI)

```
☑️ Enable Zero-Shot Classification (BART-MNLI)
```

**Warning**: 
- First run downloads 1.5GB model
- Processing becomes 2-3x slower
- Only use if accuracy is more important than speed

### **Thresholds**

Defined in `layer8_zeroshot.py`:

```python
if top_score >= 0.85:
    # High confidence - accept
elif top_score >= 0.60:
    # Moderate confidence - accept
else:
    # Low confidence - reject, pass to "Others"
```

---

## 💡 Example Use Cases

### **Case 1: Vague UPI Handle**
```
Input: "UPI-XYZ789@paytm ₹1200 debit"
L3 (Semantic): ❌ No match
L5 (Behavioral): ❌ No cluster
L8 (Zero-shot): ✅ "Shopping" (0.72)
```

### **Case 2: First-Time Merchant**
```
Input: "NewMerchant123 Payment ₹850"
L3 (Semantic): ❌ Unknown merchant
L5 (Behavioral): ❌ No history
L8 (Zero-shot): ✅ "Shopping" (0.68)
```

### **Case 3: Ambiguous Description**
```
Input: "Payment ₹5000 debit"
L3 (Semantic): ❌ Too generic
L5 (Behavioral): ❌ Amount not distinctive
L8 (Zero-shot): ⚠️ "Transfers" (0.55) → Too low, rejected
```

---

## 🎯 Advantages

✅ **Handles ANY text**: Even completely vague descriptions  
✅ **No training needed**: Pre-trained on general NLI  
✅ **Semantic understanding**: Understands context, not just keywords  
✅ **Fallback safety**: Prevents "Others/Uncategorized" overflow

---

## ⚠️ Disadvantages

❌ **Slow**: 2-3x slower than other layers  
❌ **Large model**: 1.5GB download required  
❌ **Lower accuracy**: 70-80% vs 85%+ for other layers  
❌ **Generic predictions**: May over-predict common categories

---

## 📊 Comparison

| Layer | Speed | Accuracy | When to Use |
|-------|-------|----------|-------------|
| **L0: Rules** | ⚡⚡⚡ | 95%+ | Salary, SIP, recurring |
| **L1: Canonical** | ⚡⚡⚡ | 90%+ | Known merchants |
| **L3: Semantic** | ⚡⚡ | 85%+ | Clear descriptions |
| **L5: Behavioral** | ⚡⚡ | 80%+ | Vague but has patterns |
| **L8: Zero-shot** | ⚡ | 70-80% | All other methods failed |

---

## 🔧 Implementation Details

### **Model Loading** (Lazy)

```python
def _load_model(self):
    if self.classifier is None:
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU (use 0 for GPU)
        )
```

**Only loaded when**:
- User enables zero-shot checkbox
- First transaction needs it
- Cached for subsequent uses

### **Classification Code**

```python
result = self.classifier(
    premise="UPI-MERCHANT ₹500 debit",
    candidate_labels=[
        'Food & Dining',
        'Commute/Transport',
        'Shopping',
        # ... 11 categories
    ],
    hypothesis_template="This transaction is about {}."
)

# Returns:
# {
#   'labels': ['Shopping', 'Food', 'Transport', ...],
#   'scores': [0.72, 0.15, 0.08, ...]
# }
```

---

## 📈 Expected Results

On **sample data** (200 transactions):

**Without Zero-Shot**:
```
Auto-label rate: 85%
Others/Uncategorized: 15% (30 transactions)
```

**With Zero-Shot**:
```
Auto-label rate: 90-92%
Others/Uncategorized: 8-10% (16-20 transactions)
L8 usage: 5-7% (10-14 transactions)
Processing time: +50-100% slower
```

**Trade-off**: Higher coverage vs slower processing

---

## 🎓 How to Use

### **Step 1: Install transformers**

```bash
pip install transformers==4.30.0
```

Already included in updated `requirements.txt`

### **Step 2: Enable in UI**

1. Upload your transactions
2. ☑️ Check "Enable Zero-Shot Classification (BART-MNLI)"
3. Click "Start Classification"
4. **Wait** for BART model download (first time: 2-5 min)

### **Step 3: Review Results**

Check "Layer Used" column:
- `L8: Zero-Shot (BART-MNLI)` → Transactions classified by zero-shot
- Compare confidence with other layers

---

## 🔍 Debugging

### **Check if Zero-Shot is Available**

In sidebar, look for:
```
Layers:
...
- L8: Zero-Shot (BART-MNLI) ✨
```

If not shown, `transformers` library not installed.

### **Check if Zero-Shot was Used**

In Tab 2 (Results), filter by Layer:
```
L8: Zero-Shot (BART-MNLI)
```

Should see 5-10% of transactions if enabled.

### **Check Zero-Shot Performance**

In Tab 3 (Metrics) → Layer Distribution:
```
L8: Zero-Shot (BART-MNLI) | Count: 12 | Avg Confidence: 0.68
```

---

## 💡 Best Practices

### **When to Enable**:
- ✅ High accuracy more important than speed
- ✅ Many vague/unknown transactions
- ✅ New user with limited history
- ✅ Processing can be done offline/batch

### **When to Disable**:
- ✅ Speed is critical (real-time)
- ✅ Most merchants are known
- ✅ Good behavioral patterns exist
- ✅ Can tolerate 10-15% "Others/Uncategorized"

---

## 🎯 Tuning

### **Adjust Confidence Thresholds**

In `layer8_zeroshot.py`:

```python
# More aggressive (accept more)
if top_score >= 0.55:  # Was 0.60
    return category, confidence, provenance

# More conservative (accept less)
if top_score >= 0.75:  # Was 0.60
    return category, confidence, provenance
```

### **Adjust Discount Factor**

In `layer7_classification.py`:

```python
# Trust zero-shot more
final_conf = zeroshot_conf * 0.95  # Was 0.85

# Trust zero-shot less
final_conf = zeroshot_conf * 0.75  # Was 0.85
```

---

## 📊 Model Details

**BART-Large-MNLI**:
- **Source**: HuggingFace (`facebook/bart-large-mnli`)
- **Parameters**: 406M
- **Size**: 1.5 GB
- **License**: Apache 2.0 (commercial use OK)
- **Training**: MNLI dataset (433k examples)
- **Task**: Natural Language Inference

**Download Location**:
- Windows: `C:\Users\<You>\.cache\huggingface\transformers\`
- Linux/Mac: `~/.cache/huggingface/transformers/`

---

## ✅ Summary

### **What Zero-Shot Adds**:
- ✅ Fallback for difficult cases
- ✅ +5-10% higher coverage
- ✅ Better than "Others/Uncategorized"
- ✅ Semantic understanding of any text

### **What It Costs**:
- ❌ 1.5 GB disk space
- ❌ 2-3x slower processing
- ❌ Lower confidence than other layers
- ❌ May over-predict common categories

### **When to Use**:
Use when **accuracy > speed** and you have many vague transactions.

---

## 🔗 References

- **Paper**: BART (Lewis et al., 2020)
- **Model**: https://huggingface.co/facebook/bart-large-mnli
- **MNLI Dataset**: https://cims.nyu.edu/~sbowman/multinli/
- **Zero-Shot Classification**: https://huggingface.co/tasks/zero-shot-classification

---

**Implementation Status**: ✅ Complete  
**Optional**: Yes (checkbox in UI)  
**Default**: Disabled (for speed)  
**Recommended**: Enable for final production, disable for testing

---

**Last Updated**: November 18, 2024

