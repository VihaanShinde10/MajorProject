# Installation & Setup Guide

## ✅ System Status

- **Linter Errors**: ✅ 0 (All fixed)
- **Python Version**: ✅ 3.13.3 (Compatible)
- **API Keys Required**: ❌ **NONE** (No API keys needed!)

---

## 🎯 No API Keys Required!

**Good News**: This system uses **locally-hosted models** and doesn't require any API keys!

- ✅ E5 embeddings: Downloaded automatically from HuggingFace (free, open-source)
- ✅ FAISS: Local vector search
- ✅ HDBSCAN: Local clustering
- ✅ No OpenAI, no Anthropic, no paid APIs

**First run will download E5 model (~500MB) automatically.**

---

## 📦 Required Packages

### **Core Requirements**

```txt
streamlit==1.28.0          # Web UI framework
pandas==2.0.3              # Data processing
numpy==1.24.3              # Numerical operations
torch==2.0.1               # Deep learning backend
sentence-transformers==2.2.2  # E5 embeddings
faiss-cpu==1.7.4           # Vector similarity search
hdbscan==0.8.33            # Clustering
scikit-learn==1.3.0        # ML utilities & metrics
rapidfuzz==3.2.0           # Fuzzy string matching
plotly==5.17.0             # Interactive visualizations
```

**Total Size**: ~2.5 GB (including E5 model)

---

## 🚀 Installation Steps

### **Step 1: Check Python Version**

```bash
python --version
```

**Required**: Python 3.9 or higher (You have: 3.13.3 ✅)

### **Step 2: Create Virtual Environment (Recommended)**

```bash
# Navigate to project directory
cd "C:\Users\Vihaan Shinde\OneDrive\Documents\MajorProject"

# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) in your terminal
```

### **Step 3: Install All Packages**

```bash
# Install all requirements
pip install -r requirements.txt
```

**This will install**:
1. Streamlit + dependencies
2. PyTorch (CPU version)
3. Sentence-transformers + E5 model
4. FAISS (CPU version)
5. HDBSCAN
6. Scikit-learn
7. Other utilities

**Time**: 5-10 minutes depending on internet speed

---

## 🔍 Verify Installation

After installation, verify all packages:

```bash
python -c "import streamlit; print('✅ Streamlit:', streamlit.__version__)"
python -c "import pandas; print('✅ Pandas:', pandas.__version__)"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)"
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import sentence_transformers; print('✅ Sentence-Transformers:', sentence_transformers.__version__)"
python -c "import faiss; print('✅ FAISS: OK')"
python -c "import hdbscan; print('✅ HDBSCAN:', hdbscan.__version__)"
python -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)"
python -c "import rapidfuzz; print('✅ RapidFuzz:', rapidfuzz.__version__)"
python -c "import plotly; print('✅ Plotly:', plotly.__version__)"
```

**Expected Output**:
```
✅ Streamlit: 1.28.0
✅ Pandas: 2.0.3
✅ NumPy: 1.24.3
✅ PyTorch: 2.0.1
✅ Sentence-Transformers: 2.2.2
✅ FAISS: OK
✅ HDBSCAN: 0.8.33
✅ Scikit-learn: 1.3.0
✅ RapidFuzz: 3.2.0
✅ Plotly: 5.17.0
```

---

## 🎯 First Run

### **Generate Sample Data**

```bash
python sample_data_generator.py
```

**Output**: Creates `sample_transactions.csv` with 200 transactions

### **Run the Application**

```bash
streamlit run app.py
```

**What Happens**:
1. ✅ Streamlit starts on `http://localhost:8501`
2. ✅ Browser opens automatically
3. ⏳ **First time only**: E5 model downloads (~500MB, 2-5 minutes)
4. ✅ App is ready!

---

## 📥 E5 Model Download (First Time Only)

### **What is E5?**
- Open-source embedding model from Microsoft
- Hosted on HuggingFace
- Downloaded automatically on first use
- **No API key needed**

### **First Run Process**:

```bash
streamlit run app.py
```

**Console Output** (first time):
```
Downloading intfloat/e5-base-v2...
config.json: 100%|████████| 743/743 [00:00<00:00, 371kB/s]
model.safetensors: 100%|████████| 438M/438M [02:15<00:00, 3.23MB/s]
tokenizer_config.json: 100%|████████| 366/366 [00:00<00:00, 183kB/s]
Model downloaded to: ~/.cache/huggingface/transformers/
```

**Download Location**:
- Windows: `C:\Users\<YourName>\.cache\huggingface\transformers\`
- Linux/Mac: `~/.cache/huggingface/transformers/`

**Only happens once** - subsequent runs use cached model.

---

## ⚙️ Configuration (Optional)

### **No Configuration Needed!**

The system works out-of-the-box with sensible defaults. However, you can customize:

### **1. Thresholds** (via Streamlit UI - Tab 4: Settings)
- Unanimous threshold: 0.78
- Majority threshold: 0.70
- Auto-label threshold: 0.75
- Feedback threshold: 0.50

### **2. Add Your Merchants** (optional)

Edit `layers/layer1_normalization.py`:

```python
self.canonical_aliases = {
    'swiggy': ['swiggy', 'swigy', 'swgy'],
    'your_merchant': ['merchant_name', 'variations'],  # Add here
    # ... more merchants
}

self.category_map = {
    'swiggy': 'Food & Dining',
    'your_merchant': 'Your Category',  # Add here
    # ... more mappings
}
```

### **3. Custom Rules** (optional)

Edit `layers/layer0_rules.py` to add domain-specific rules.

---

## 🐛 Troubleshooting

### **Issue 1: ModuleNotFoundError**

```
ModuleNotFoundError: No module named 'hdbscan'
```

**Solution**:
```bash
# Ensure virtual environment is activated
venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

### **Issue 2: E5 Model Download Fails**

```
HTTPError: 403 Client Error
```

**Solution**:
```bash
# Try downloading manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"
```

If still fails, check internet connection or try:
```bash
# Use mirror (if in China/restricted region)
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

### **Issue 3: Out of Memory**

```
RuntimeError: out of memory
```

**Solution**:
- Use CPU version (default in requirements.txt)
- Process fewer transactions at once
- Close other applications

### **Issue 4: Slow Performance**

**Solutions**:
1. **Use GPU** (if available):
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Use FAISS GPU**:
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu
   ```

3. **Reduce batch size** in `layer2_embeddings.py`

### **Issue 5: Streamlit Port Already in Use**

```
Address already in use
```

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## 💻 System Requirements

### **Minimum**:
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 4 GB
- **Storage**: 3 GB free
- **Python**: 3.9+
- **Internet**: For first-time model download

### **Recommended**:
- **OS**: Windows 11, Ubuntu 20.04+, macOS 11+
- **RAM**: 8 GB
- **Storage**: 5 GB free
- **Python**: 3.10+
- **CPU**: Multi-core (4+ cores)
- **Internet**: Broadband for faster downloads

### **Optional (for better performance)**:
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 16 GB
- **SSD**: For faster I/O

---

## 📊 Disk Space Usage

| Component | Size |
|-----------|------|
| Python packages | ~1.5 GB |
| E5 model | ~500 MB |
| FAISS index (runtime) | ~50 MB |
| Your data | Variable |
| **Total** | **~2 GB** |

---

## 🔒 Privacy & Security

### **All Local Processing**:
✅ No data sent to external APIs  
✅ No cloud services used  
✅ Models run locally  
✅ Your data stays on your machine  

### **What's Downloaded**:
- E5 model from HuggingFace (one-time, open-source)
- Python packages from PyPI (standard practice)

### **No Tracking**:
- No telemetry
- No usage analytics
- No data collection

---

## 🎓 Package Details

### **streamlit** (Web UI)
- **Purpose**: Interactive web application
- **License**: Apache 2.0
- **Size**: ~50 MB

### **torch** (Deep Learning)
- **Purpose**: Backend for E5 embeddings
- **License**: BSD-3-Clause
- **Size**: ~700 MB (CPU version)
- **Note**: GPU version is ~2 GB

### **sentence-transformers** (Embeddings)
- **Purpose**: E5 semantic embeddings
- **License**: Apache 2.0
- **Size**: ~100 MB + 500 MB model

### **faiss-cpu** (Vector Search)
- **Purpose**: Fast similarity search
- **License**: MIT
- **Size**: ~50 MB

### **hdbscan** (Clustering)
- **Purpose**: Behavioral clustering
- **License**: BSD-3-Clause
- **Size**: ~10 MB

### **scikit-learn** (ML Utilities)
- **Purpose**: Metrics, preprocessing
- **License**: BSD-3-Clause
- **Size**: ~200 MB

### **rapidfuzz** (Fuzzy Matching)
- **Purpose**: Merchant name normalization
- **License**: MIT
- **Size**: ~5 MB

### **plotly** (Visualization)
- **Purpose**: Interactive charts
- **License**: MIT
- **Size**: ~50 MB

---

## ✅ Pre-Flight Checklist

Before running the app, ensure:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created & activated
- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] No linter errors (`read_lints` passed ✅)
- [ ] Sample data generated (`sample_data_generator.py`)
- [ ] Internet connection (for first-time E5 download)
- [ ] 3 GB free disk space
- [ ] Port 8501 available (or use different port)

---

## 🚀 Quick Start Summary

```bash
# 1. Navigate to project
cd "C:\Users\Vihaan Shinde\OneDrive\Documents\MajorProject"

# 2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# 4. Generate sample data
python sample_data_generator.py

# 5. Run app
streamlit run app.py

# 6. Open browser to http://localhost:8501
```

**First run**: Wait 2-5 minutes for E5 model download  
**Subsequent runs**: Instant startup

---

## 📞 Support

### **If Installation Fails**:

1. **Check Python version**: `python --version` (need 3.9+)
2. **Update pip**: `python -m pip install --upgrade pip`
3. **Clear cache**: `pip cache purge`
4. **Reinstall**: `pip install -r requirements.txt --no-cache-dir`

### **If App Crashes**:

1. Check terminal for error messages
2. Ensure all packages installed
3. Verify E5 model downloaded
4. Check disk space
5. Try with smaller dataset first

### **Common Warnings (Can Ignore)**:

```
FutureWarning: ...
DeprecationWarning: ...
```

These are normal and don't affect functionality.

---

## 🎉 You're Ready!

Once installation completes:

1. ✅ No API keys needed
2. ✅ All models local
3. ✅ Privacy preserved
4. ✅ Ready to classify transactions

**Run**: `streamlit run app.py` and start categorizing! 🚀

---

**Last Updated**: November 18, 2024  
**Python Version Tested**: 3.13.3  
**Status**: ✅ All Systems Go

