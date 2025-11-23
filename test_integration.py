"""
Quick test to verify enhanced components are loaded.
Run this to check if the system is using enhanced features.
"""

import sys
import pandas as pd
from datetime import datetime

# Fix encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("TESTING ENHANCED COMPONENTS INTEGRATION")
print("="*80)

# Test 1: Check imports
print("\n1. Testing Imports...")
try:
    from layers.layer4_behavioral_features_enhanced import EnhancedBehavioralFeatureExtractor
    print("   ✅ Enhanced Behavioral Features available")
    enhanced_features = True
except ImportError as e:
    print(f"   ❌ Enhanced Behavioral Features NOT available: {e}")
    enhanced_features = False

try:
    from layers.layer3_semantic_search_attention import SemanticSearcherWithAttention
    print("   ✅ Attention-based Semantic Search available")
    attention_search = True
except ImportError as e:
    print(f"   ❌ Attention-based Semantic Search NOT available: {e}")
    attention_search = False

try:
    from layers.gating_trainer import GatingTrainer
    print("   ✅ Gating Trainer available")
    gating_trainer = True
except ImportError as e:
    print(f"   ❌ Gating Trainer NOT available: {e}")
    gating_trainer = False

try:
    from layers.ensemble_classifier import EnsembleClassifier
    print("   ✅ Ensemble Classifier available")
    ensemble = True
except ImportError as e:
    print(f"   ❌ Ensemble Classifier NOT available: {e}")
    ensemble = False

try:
    from evaluation.ablation_baselines import AblationBaselines
    print("   ✅ Ablation Baselines available")
    ablation = True
except ImportError as e:
    print(f"   ❌ Ablation Baselines NOT available: {e}")
    ablation = False

try:
    from evaluation.cross_validation import TransactionCrossValidator
    print("   ✅ Cross-Validation available")
    cv = True
except ImportError as e:
    print(f"   ❌ Cross-Validation NOT available: {e}")
    cv = False

try:
    from evaluation.advanced_metrics import AdvancedMetricsCalculator
    print("   ✅ Advanced Metrics available")
    metrics = True
except ImportError as e:
    print(f"   ❌ Advanced Metrics NOT available: {e}")
    metrics = False

# Test 2: Check feature extraction
print("\n2. Testing Feature Extraction...")
if enhanced_features:
    from layers.layer4_behavioral_features_enhanced import EnhancedBehavioralFeatureExtractor
    extractor = EnhancedBehavioralFeatureExtractor()
    
    # Create dummy transaction
    txn = pd.Series({
        'date': datetime.now(),
        'amount': 500.0,
        'description': 'Test transaction',
        'type': 'debit',
        'merchant': 'Test Merchant'
    })
    
    # Create dummy history
    history = pd.DataFrame()
    
    # Extract features
    features = extractor.extract(txn, history)
    
    print(f"   ✅ Extracted {len(features)} features")
    print(f"   Feature names (first 10): {list(features.keys())[:10]}")
    
    # Check for enhanced features
    enhanced_feature_names = ['spending_velocity_7d', 'merchant_degree', 'hint_food']
    found_enhanced = [f for f in enhanced_feature_names if f in features]
    
    if found_enhanced:
        print(f"   ✅ Enhanced features detected: {found_enhanced}")
    else:
        print(f"   ⚠️ Enhanced features NOT found in output")
else:
    print("   ⏭️ Skipped (enhanced features not available)")

# Test 3: Check gating network
print("\n3. Testing Gating Network...")
try:
    from layers.layer6_gating import GatingController
    import os
    
    controller = GatingController(
        model_path='models/gating_trained.pt',
        use_trained_model=True
    )
    
    if os.path.exists('models/gating_trained.pt'):
        print("   ✅ Trained gating model found")
        if controller.model_loaded:
            print("   ✅ Trained model loaded successfully")
        else:
            print("   ⚠️ Model file exists but not loaded")
    else:
        print("   ℹ️ No trained model (using heuristics)")
    
    # Test alpha computation
    alpha = controller.compute_alpha(
        text_confidence=0.8,
        token_count=5,
        is_generic_text=False,
        recurrence_confidence=0.6,
        cluster_density=0.7,
        user_txn_count=100
    )
    print(f"   ✅ Alpha computation works: α = {alpha:.3f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Check calibrator
print("\n4. Testing Confidence Calibrator...")
try:
    from layers.ensemble_classifier import EnsembleClassifier
    import os
    
    ensemble = EnsembleClassifier()
    
    if os.path.exists('models/confidence_calibrator.pkl'):
        ensemble.load_calibrator('models/confidence_calibrator.pkl')
        print("   ✅ Calibrator loaded successfully")
        
        # Test calibration
        calibrated = ensemble.apply_calibration(0.75)
        print(f"   ✅ Calibration works: 0.75 → {calibrated:.3f}")
    else:
        print("   ℹ️ No calibrator file (raw confidence will be used)")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

components = {
    "Enhanced Behavioral Features (50+)": enhanced_features,
    "Attention-based Semantic Search": attention_search,
    "Gating Network Trainer": gating_trainer,
    "Ensemble Classifier": ensemble,
    "Ablation Study": ablation,
    "Cross-Validation": cv,
    "Advanced Metrics": metrics
}

# Convert boolean values
components = {k: bool(v) for k, v in components.items()}

available = sum(components.values())
total = len(components)

print(f"\n✅ {available}/{total} enhanced components available")

if available == total:
    print("\n🎉 ALL ENHANCED COMPONENTS AVAILABLE!")
    print("   Your system is using the full robust implementation.")
else:
    print(f"\n⚠️ {total - available} components missing")
    print("   System will work but with reduced capabilities.")

print("\nNext steps:")
if not enhanced_features or not attention_search:
    print("  - Check that all new files are in the correct directories")
    print("  - Verify file names match exactly")
    
import os
if not os.path.exists('models/gating_trained.pt'):
    print("  - Train models: python train_system.py training_data.csv")
    
print("\n" + "="*80)

