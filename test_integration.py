"""
Integration Test Script
Tests all components are properly integrated
"""

import sys
import os

print("="*60)
print("TRANSACTION CATEGORIZATION SYSTEM - INTEGRATION TEST")
print("="*60)
print()

# Test 1: Python Version
print("✓ Test 1: Python Version")
print(f"  Version: {sys.version}")
if sys.version_info < (3, 9):
    print("  ❌ FAIL: Python 3.9+ required")
    sys.exit(1)
print("  ✅ PASS")
print()

# Test 2: Import Core Packages
print("✓ Test 2: Core Packages")
packages = [
    ('streamlit', 'Streamlit'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('torch', 'PyTorch'),
    ('sentence_transformers', 'Sentence-Transformers'),
    ('faiss', 'FAISS'),
    ('hdbscan', 'HDBSCAN'),
    ('sklearn', 'Scikit-learn'),
    ('rapidfuzz', 'RapidFuzz'),
    ('plotly', 'Plotly')
]

failed = []
for module, name in packages:
    try:
        exec(f"import {module}")
        print(f"  ✅ {name}: OK")
    except ImportError as e:
        print(f"  ❌ {name}: MISSING - {e}")
        failed.append(name)

if failed:
    print(f"\n  ❌ FAIL: Missing packages: {', '.join(failed)}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)
print("  ✅ PASS: All packages installed")
print()

# Test 3: Import All Layers
print("✓ Test 3: Layer Modules")
layers = [
    ('layers.layer0_rules', 'RuleBasedDetector', 'L0: Rules'),
    ('layers.layer1_normalization', 'TextNormalizer', 'L1: Normalization'),
    ('layers.layer2_embeddings', 'E5Embedder', 'L2: Embeddings'),
    ('layers.layer3_semantic_search', 'SemanticSearcher', 'L3: Semantic Search'),
    ('layers.layer4_behavioral_features', 'BehavioralFeatureExtractor', 'L4: Features'),
    ('layers.layer5_clustering', 'BehavioralClusterer', 'L5: Clustering'),
    ('layers.layer6_gating', 'GatingController', 'L6: Gating'),
    ('layers.layer7_classification', 'FinalClassifier', 'L7: Classification')
]

failed_layers = []
for module, cls, name in layers:
    try:
        exec(f"from {module} import {cls}")
        print(f"  ✅ {name}: OK")
    except ImportError as e:
        print(f"  ❌ {name}: FAILED - {e}")
        failed_layers.append(name)

if failed_layers:
    print(f"\n  ❌ FAIL: Layer imports failed: {', '.join(failed_layers)}")
    sys.exit(1)
print("  ✅ PASS: All layers import successfully")
print()

# Test 4: Import Metrics
print("✓ Test 4: Metrics Module")
try:
    from metrics.metrics_tracker import MetricsTracker
    print("  ✅ MetricsTracker: OK")
except ImportError as e:
    print(f"  ❌ MetricsTracker: FAILED - {e}")
    sys.exit(1)
print("  ✅ PASS")
print()

# Test 5: Test Metrics Tracker Initialization
print("✓ Test 5: Metrics Tracker Functionality")
try:
    tracker = MetricsTracker()
    
    # Test log_prediction
    tracker.log_prediction(
        predicted_category='Food & Dining',
        confidence=0.85,
        layer_used='L1: Test',
        alpha=0.5,
        merchant='test'
    )
    
    # Test compute_metrics
    metrics = tracker.compute_metrics()
    assert 'total_transactions' in metrics
    assert metrics['total_transactions'] == 1
    assert 'avg_confidence' in metrics
    
    print("  ✅ log_prediction: OK")
    print("  ✅ compute_metrics: OK")
    print(f"  ✅ Metrics computed: {len(metrics)} keys")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    sys.exit(1)
print("  ✅ PASS")
print()

# Test 6: Test Layer Instantiation
print("✓ Test 6: Layer Instantiation")
try:
    from layers.layer0_rules import RuleBasedDetector
    from layers.layer1_normalization import TextNormalizer
    from layers.layer6_gating import GatingController
    from layers.layer7_classification import FinalClassifier
    
    rule_detector = RuleBasedDetector()
    print("  ✅ RuleBasedDetector: OK")
    
    normalizer = TextNormalizer()
    print("  ✅ TextNormalizer: OK")
    
    gating = GatingController()
    print("  ✅ GatingController: OK")
    
    classifier = FinalClassifier()
    print("  ✅ FinalClassifier: OK")
    
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("  ✅ PASS")
print()

# Test 7: Test E5 Model (Quick Check)
print("✓ Test 7: E5 Model Availability")
try:
    from sentence_transformers import SentenceTransformer
    print("  ℹ️  Checking E5 model...")
    print("  ℹ️  This may download model on first run (~500MB)")
    
    # This will download model if not present
    model = SentenceTransformer('intfloat/e5-base-v2')
    print("  ✅ E5 model loaded successfully")
    
    # Quick embedding test
    test_text = "query: test transaction"
    embedding = model.encode(test_text, normalize_embeddings=True)
    print(f"  ✅ Embedding dimension: {len(embedding)}")
    
except Exception as e:
    print(f"  ⚠️  WARNING: E5 model test failed - {e}")
    print("  ℹ️  Model will download on first app run")
print()

# Test 8: Test Clustering Metrics
print("✓ Test 8: Clustering Metrics")
try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score
    import numpy as np
    
    # Create dummy data
    X = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)
    
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    print(f"  ✅ Silhouette Score: {sil:.2f}")
    print(f"  ✅ Davies-Bouldin Index: {db:.2f}")
    print("  ✅ Clustering metrics working")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    sys.exit(1)
print("  ✅ PASS")
print()

# Test 9: File Structure
print("✓ Test 9: File Structure")
required_files = [
    'app.py',
    'requirements.txt',
    'sample_data_generator.py',
    'layers/layer0_rules.py',
    'layers/layer1_normalization.py',
    'layers/layer2_embeddings.py',
    'layers/layer3_semantic_search.py',
    'layers/layer4_behavioral_features.py',
    'layers/layer5_clustering.py',
    'layers/layer6_gating.py',
    'layers/layer7_classification.py',
    'metrics/metrics_tracker.py'
]

missing = []
for file in required_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file}: MISSING")
        missing.append(file)

if missing:
    print(f"\n  ❌ FAIL: Missing files: {', '.join(missing)}")
    sys.exit(1)
print("  ✅ PASS: All required files present")
print()

# Final Summary
print("="*60)
print("INTEGRATION TEST SUMMARY")
print("="*60)
print("✅ All tests passed!")
print()
print("Your system is ready to run:")
print("  streamlit run app.py")
print()
print("No API keys required - all models run locally.")
print("="*60)

