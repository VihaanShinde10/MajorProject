# Installation Options

## 📦 Three Ways to Install

We provide **three requirements files** to handle different scenarios and avoid version conflicts.

---

## Option 1: Standard Installation (Recommended)

**File**: `requirements.txt`

**Use When**:
- ✅ Fresh installation
- ✅ Python 3.9-3.11
- ✅ Standard system
- ✅ No existing packages

```bash
pip install -r requirements.txt
```

**What it does**:
- Installs packages with minimum version requirements
- Allows pip to resolve compatible versions
- Most flexible, handles most systems

---

## Option 2: Minimal Installation

**File**: `requirements-minimal.txt`

**Use When**:
- ⚠️ Version conflicts with `requirements.txt`
- ⚠️ Want latest versions of everything
- ⚠️ Willing to troubleshoot

```bash
pip install -r requirements-minimal.txt
```

**What it does**:
- Installs latest compatible versions
- No minimum version constraints
- Lets pip figure everything out

---

## Option 3: Windows-Optimized

**File**: `requirements-windows.txt`

**Use When**:
- 🪟 Windows 10/11
- ⚠️ Having installation issues on Windows
- ⚠️ PyTorch or transformers failing

```bash
pip install -r requirements-windows.txt
```

**What it does**:
- Conservative version ranges
- Tested on Windows
- Avoids known Windows issues

---

## 🔍 Version Explanations

### **NumPy: `<2.0.0`**
**Why**: NumPy 2.0 breaks compatibility with many libraries
- **Issue**: `AttributeError: module 'numpy' has no attribute 'float_'`
- **Fix**: Restrict to NumPy 1.x
- **Safe**: NumPy 1.24+ is stable and fast

### **Streamlit: `>=1.28.0`**
**Why**: Earlier versions missing features we use
- **Features**: `st.tabs()`, `hide_index` in dataframe
- **Safe**: 1.28+ is stable

### **Torch: `>=2.0.0`**
**Why**: PyTorch 2.0+ has better performance
- **Performance**: ~30% faster inference
- **Compatibility**: Works with all sentence-transformers versions
- **Note**: CPU version automatically selected

### **Transformers: `>=4.30.0`**
**Why**: BART-MNLI requires 4.30+
- **Features**: Better pipeline API
- **Note**: Optional (only if using zero-shot)

### **Others: Flexible**
- All other packages use `>=` for maximum compatibility

---

## 🐛 Troubleshooting Installation

### **Issue 1: NumPy Version Conflict**

```
ERROR: numpy 2.0.0 is incompatible with pandas 2.0.3
```

**Solution**:
```bash
pip install "numpy<2.0.0" --force-reinstall
pip install -r requirements.txt
```

### **Issue 2: Torch Installation Fails**

```
ERROR: Could not find a version that satisfies torch
```

**Solution** (Windows):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-deps
```

**Solution** (Linux/Mac):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### **Issue 3: HDBSCAN Build Fails**

```
ERROR: Failed building wheel for hdbscan
```

**Solution** (Windows):
```bash
# Install Visual C++ Build Tools first
# Then:
pip install hdbscan --no-cache-dir
```

**Solution** (Linux):
```bash
sudo apt-get install python3-dev
pip install hdbscan
```

### **Issue 4: FAISS Installation Fails**

```
ERROR: Could not find a version that satisfies faiss-cpu
```

**Solution**:
```bash
# Try conda instead (if available)
conda install -c conda-forge faiss-cpu

# OR use alternative channel
pip install faiss-cpu --extra-index-url https://pypi.anaconda.org/conda-forge/simple
```

### **Issue 5: Transformers Version Conflict**

```
ERROR: transformers 5.0.0 breaks with sentence-transformers
```

**Solution**:
```bash
pip install "transformers>=4.30.0,<5.0.0"
```

---

## 🎯 Step-by-Step Installation (Safest)

If you're having issues, install packages **one by one**:

```bash
# 1. Core packages first
pip install streamlit
pip install pandas
pip install "numpy<2.0.0"

# 2. ML packages
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn

# 3. NLP packages
pip install sentence-transformers
pip install transformers  # Optional for zero-shot

# 4. Specialized packages
pip install faiss-cpu
pip install hdbscan
pip install rapidfuzz
pip install plotly

# 5. Verify
python -c "import streamlit, pandas, numpy, torch, sentence_transformers, faiss, hdbscan, sklearn, rapidfuzz, plotly; print('✅ All packages installed')"
```

---

## 🔍 Version Check Commands

Check what versions you have installed:

```bash
# Check individual packages
pip show streamlit
pip show numpy
pip show torch
pip show transformers

# Check all at once
pip list | grep -E "streamlit|pandas|numpy|torch|transformers|faiss|hdbscan|scikit-learn|rapidfuzz|plotly"
```

---

## 🆕 Update Packages

If you installed earlier and want to update:

```bash
# Update all to latest compatible versions
pip install -r requirements.txt --upgrade

# Update specific package
pip install --upgrade streamlit
pip install --upgrade transformers
```

---

## 🐍 Python Version Compatibility

| Python Version | Status | Notes |
|----------------|--------|-------|
| **3.9** | ✅ Fully supported | Recommended minimum |
| **3.10** | ✅ Fully supported | Recommended |
| **3.11** | ✅ Fully supported | Best performance |
| **3.12** | ⚠️ Mostly works | Some packages may need building |
| **3.13** | ⚠️ Experimental | You have this - should work |
| **3.8** | ❌ Not supported | Too old for some packages |

**Your Version**: Python 3.13.3 ✅ (Should work fine)

---

## 🌍 Platform-Specific Notes

### **Windows**
- Use `requirements-windows.txt` if issues
- May need Visual C++ Build Tools for HDBSCAN
- PyTorch CPU version works best

### **Linux**
- Use `requirements.txt` (standard)
- May need: `sudo apt-get install python3-dev`
- FAISS and HDBSCAN compile from source

### **macOS**
- Use `requirements.txt` (standard)
- M1/M2 Macs: Use native Python, not Rosetta
- PyTorch has ARM64 support

---

## 📊 Disk Space Requirements

| Package | Approximate Size |
|---------|-----------------|
| streamlit | ~50 MB |
| pandas + numpy | ~150 MB |
| torch (CPU) | ~700 MB |
| sentence-transformers | ~100 MB |
| transformers | ~200 MB |
| faiss-cpu | ~50 MB |
| hdbscan | ~10 MB |
| scikit-learn | ~200 MB |
| Others | ~50 MB |
| **Total** | **~1.5 GB** |
| **+ E5 model** | **+500 MB** |
| **+ BART model** | **+1.5 GB** (optional) |
| **Grand Total** | **~3.5 GB** (with all models) |

---

## ✅ Installation Success Checklist

After installation, verify:

```bash
# Run integration test
python test_integration.py
```

**Should see**:
```
✅ Test 1: Python Version PASS
✅ Test 2: Core Packages PASS
✅ Test 3: Layer Modules PASS
✅ Test 4: Metrics Module PASS
✅ Test 5: Metrics Functionality PASS
✅ Test 6: Layer Instantiation PASS
✅ Test 7: E5 Model PASS
✅ Test 8: Clustering Metrics PASS
✅ Test 9: File Structure PASS

✅ All tests passed!
```

---

## 🎯 Recommended Installation Path

**For most users**:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install packages
pip install -r requirements.txt

# 5. Verify
python test_integration.py

# 6. Run app
streamlit run app.py
```

---

## 🆘 Still Having Issues?

### **Option 1: Use Conda**

```bash
conda create -n spendwise python=3.11
conda activate spendwise
conda install -c conda-forge streamlit pandas numpy scikit-learn plotly
pip install torch sentence-transformers transformers hdbscan faiss-cpu rapidfuzz
```

### **Option 2: Docker** (Future)

```bash
# Will be provided if needed
docker build -t spendwise .
docker run -p 8501:8501 spendwise
```

### **Option 3: Cloud Deployment**

```bash
# Deploy to Streamlit Cloud (handles dependencies)
# No local installation needed
```

---

## 📝 Summary

**Three options**:
1. ✅ `requirements.txt` - Standard, recommended
2. ⚠️ `requirements-minimal.txt` - If version conflicts
3. 🪟 `requirements-windows.txt` - Windows-specific

**Key points**:
- All use **flexible version ranges** (`>=`)
- NumPy pinned to `<2.0.0` (important!)
- No exact versions (no `==`)
- Maximum compatibility

**Your system**: Python 3.13.3 on Windows ✅
**Recommended**: Try `requirements.txt` first, use `requirements-windows.txt` if issues

---

**Last Updated**: November 18, 2024  
**Status**: ✅ Flexible versions, no pinning

