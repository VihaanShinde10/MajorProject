import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add layers to path
sys.path.insert(0, os.path.dirname(__file__))

from layers.layer0_rules import RuleBasedDetector
from layers.layer1_normalization import TextNormalizer
from layers.layer2_embeddings import E5Embedder
from layers.layer3_semantic_search import SemanticSearcher
from layers.layer4_behavioral_features import BehavioralFeatureExtractor
from layers.layer5_clustering import BehavioralClusterer
from layers.layer6_gating import GatingController
from layers.layer7_classification import FinalClassifier
from metrics.metrics_tracker import MetricsTracker

# Optional: Zero-shot classifier (BART-MNLI)
try:
    from layers.layer8_zeroshot import ZeroShotClassifier
    ZEROSHOT_AVAILABLE = True
except ImportError:
    ZEROSHOT_AVAILABLE = False
    ZeroShotClassifier = None

# Page config
st.set_page_config(
    page_title="Transaction Categorization System",
    page_icon="💳",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.transactions = None
    st.session_state.results = []
    st.session_state.metrics_tracker = MetricsTracker()
    st.session_state.embedder = None
    st.session_state.semantic_searcher = None
    st.session_state.clusterer = None

# Sidebar
st.sidebar.title("💳 Transaction Categorization")
st.sidebar.markdown("### 7-Layer Hybrid System")
layers_list = """
**Layers:**
- L0: Rule-Based Detection
- L1: Text Normalization
- L2: E5 Embeddings
- L3: FAISS Semantic Search
- L4: Behavioral Features
- L5: HDBSCAN Clustering
- L6: Gating Network
- L7: Final Classification
"""

if ZEROSHOT_AVAILABLE:
    layers_list += "- L8: Zero-Shot (BART-MNLI) ✨\n"

st.sidebar.markdown(layers_list)

# Main app
st.title("🔍 Transaction Categorization System")
st.markdown("**Unsupervised Hybrid Semantic-Behavioral Categorization**")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Classify", "📊 Results", "📈 Metrics", "⚙️ Settings"])

with tab1:
    st.header("Upload Transactions")
    
    # Zero-shot option
    use_zeroshot = False
    if ZEROSHOT_AVAILABLE:
        use_zeroshot = st.checkbox(
            "🔮 Enable Zero-Shot Classification (BART-MNLI)",
            value=False,
            help="Fallback layer for difficult cases. Warning: Slower processing (~2-3x), downloads additional 1.5GB model."
        )
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Standardize column names to match expected format
        column_mapping = {
            'Transaction_Date': 'date',
            'Description': 'description',
            'Debit': 'debit',
            'Credit': 'credit',
            'Transaction_Mode': 'mode',
            'DR/CR_Indicator': 'type'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)
        
        # Create amount column from Debit/Credit
        if 'debit' in df.columns and 'credit' in df.columns:
            df['amount'] = df['debit'].fillna(0) + df['credit'].fillna(0)
        
        # Convert type from DR/CR to debit/credit
        if 'type' in df.columns:
            df['type'] = df['type'].str.lower().map({'dr': 'debit', 'cr': 'credit'})
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M', errors='coerce')
        
        st.session_state.transactions = df
        
        st.success(f"✅ Loaded {len(df)} transactions")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Required columns check
        required_cols = ['date', 'amount', 'description', 'type']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
        else:
            if st.button("🚀 Start Classification", type="primary"):
                # Progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Initializing layers..."):
                    # Initialize all layers
                    status_text.text("Loading layers...")
                    rule_detector = RuleBasedDetector()
                    normalizer = TextNormalizer()
                    embedder = E5Embedder()
                    semantic_searcher = SemanticSearcher()
                    feature_extractor = BehavioralFeatureExtractor()
                    clusterer = BehavioralClusterer()
                    gating_controller = GatingController()
                    final_classifier = FinalClassifier()
                    
                    # Initialize zero-shot if enabled
                    zeroshot_classifier = None
                    if use_zeroshot and ZEROSHOT_AVAILABLE:
                        status_text.text("Loading BART-MNLI model (first time: ~1.5GB download)...")
                        zeroshot_classifier = ZeroShotClassifier()
                        status_text.text("✅ Zero-shot classifier loaded")
                    
                    st.session_state.embedder = embedder
                    
                    # Add merchant column if not present
                    if 'merchant' not in df.columns:
                        df['merchant'] = df['description']
                    
                    results = []
                    
                    # Phase 1: Build semantic index from initial data
                    status_text.text("Phase 1: Building semantic index...")
                    if len(df) > 50:  # Only if enough data
                        sample_df = df.head(100)
                        normalized_texts = []
                        
                        for idx, row in sample_df.iterrows():
                            text = str(row.get('description', ''))
                            normalized, _ = normalizer.normalize(text)
                            normalized_texts.append(normalized)
                        
                        # Get embeddings
                        embeddings = embedder.embed_batch(normalized_texts)
                        
                        # Use normalized names as labels (placeholder)
                        labels = normalized_texts
                        metadata = [{'idx': i} for i in range(len(normalized_texts))]
                        
                        semantic_searcher.build_index(embeddings, labels, metadata)
                        st.session_state.semantic_searcher = semantic_searcher
                    
                    # Phase 2: Extract behavioral features and cluster
                    status_text.text("Phase 2: Extracting behavioral features...")
                    features_list = []
                    features_dict = {}  # Map df index to features
                    
                    for idx, row in df.iterrows():
                        history = df[df.index < idx]
                        features = feature_extractor.extract(row, history)
                        features_list.append(features)
                        features_dict[idx] = features  # Store with df index as key
                    
                    features_df = pd.DataFrame(features_list)
                    
                    # Cluster if enough data
                    if len(df) > 20:
                        status_text.text("Phase 3: Clustering transactions...")
                        cluster_ids = clusterer.fit(features_df)
                        st.session_state.clusterer = clusterer
                        
                        # Store clustering data for metrics
                        st.session_state.metrics_tracker.set_clustering_data(
                            feature_vectors=clusterer.feature_vectors,
                            cluster_labels=clusterer.clusterer.labels_
                        )
                    
                    # Phase 4: Classify each transaction
                    status_text.text("Phase 4: Classifying transactions...")
                    
                    for idx, row in df.iterrows():
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        
                        history = df[df.index < idx]
                        
                        # Layer 0: Rules
                        rule_result = rule_detector.detect(row, history)
                        
                        # Layer 1: Normalization
                        text = str(row.get('description', ''))
                        normalized_text, norm_metadata = normalizer.normalize(text)
                        
                        # Check if canonical match exists
                        if norm_metadata.get('canonical_match') and norm_metadata.get('canonical_confidence', 0) > 0.9:
                            canonical = norm_metadata['canonical_match']
                            category = normalizer.category_map.get(canonical, 'Others/Uncategorized')
                            
                            result = {
                                'transaction_id': idx,
                                'original_description': text,
                                'normalized_text': normalized_text,
                                'category': category,
                                'confidence': norm_metadata['canonical_confidence'],
                                'layer_used': 'L1: Canonical Match',
                                'reason': f"Matched to known merchant: {canonical}",
                                'should_prompt': False
                            }
                            results.append(result)
                            st.session_state.metrics_tracker.log_prediction(
                                category, 
                                norm_metadata['canonical_confidence'],
                                'L1: Canonical Match',
                                alpha=None,
                                merchant=canonical,
                                true_category=row.get('true_category', None)
                            )
                            continue
                        
                        # Layer 2 & 3: Embeddings + Semantic Search
                        embedding = embedder.embed(normalized_text, row.get('type', ''))
                        
                        semantic_result = (None, 0.0, {})
                        if st.session_state.semantic_searcher and st.session_state.semantic_searcher.index:
                            semantic_result = semantic_searcher.search(embedding)
                        
                        # Layer 4 & 5: Behavioral features + Clustering
                        features = features_dict[idx]
                        behavioral_result = (None, 0.0, {})
                        
                        if st.session_state.clusterer and st.session_state.clusterer.clusterer:
                            behavioral_result = clusterer.predict(features)
                        
                        # Layer 6: Gating
                        text_conf = semantic_result[1]
                        behavior_conf = behavioral_result[1]
                        
                        alpha = gating_controller.compute_alpha(
                            text_confidence=text_conf,
                            token_count=norm_metadata.get('token_count', 0),
                            is_generic_text=(text_conf < 0.5),
                            recurrence_confidence=features.get('recurrence_confidence', 0.0),
                            cluster_density=behavior_conf,
                            user_txn_count=len(history)
                        )
                        
                        # Layer 8: Zero-shot (optional, only if enabled and needed)
                        zeroshot_result = (None, 0.0, {})
                        if zeroshot_classifier and (not semantic_result[0] and not behavioral_result[0]):
                            # Only use zero-shot if other methods failed
                            zeroshot_result = zeroshot_classifier.classify(
                                description=text,
                                merchant=row.get('merchant', ''),
                                amount=row['amount'],
                                txn_type=row.get('type', '')
                            )
                        
                        # Layer 7: Final Classification
                        classification = final_classifier.classify(
                            rule_result,
                            semantic_result,
                            behavioral_result,
                            alpha,
                            text_conf,
                            behavior_conf,
                            zeroshot_result
                        )
                        
                        result = {
                            'transaction_id': idx,
                            'original_description': text,
                            'normalized_text': normalized_text,
                            'category': classification.category,
                            'confidence': classification.confidence,
                            'layer_used': classification.layer_used,
                            'reason': classification.reason,
                            'should_prompt': classification.should_prompt,
                            'alpha': alpha
                        }
                        
                        results.append(result)
                        
                        st.session_state.metrics_tracker.log_prediction(
                            classification.category,
                            classification.confidence,
                            classification.layer_used,
                            alpha=alpha,
                            merchant=row.get('merchant', ''),
                            true_category=row.get('true_category', None)
                        )
                    
                    st.session_state.results = results
                    st.session_state.initialized = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Classification complete!")
                    
                    st.success(f"✅ Classified {len(results)} transactions")
                    st.balloons()

with tab2:
    st.header("Classification Results")
    
    if st.session_state.results:
        results_df = pd.DataFrame(st.session_state.results)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_conf = results_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
        
        with col2:
            auto_rate = (results_df['confidence'] >= 0.75).sum() / len(results_df)
            st.metric("Auto-Label Rate", f"{auto_rate:.2%}")
        
        with col3:
            unique_cats = results_df['category'].nunique()
            st.metric("Categories Found", unique_cats)
        
        with col4:
            low_conf = (results_df['confidence'] < 0.50).sum()
            st.metric("Low Confidence", low_conf)
        
        # Category distribution
        st.subheader("Category Distribution")
        category_counts = results_df['category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Transactions by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Layer distribution
        st.subheader("Layer Usage Distribution")
        layer_counts = results_df['layer_used'].value_counts()
        
        fig = px.bar(
            x=layer_counts.index,
            y=layer_counts.values,
            labels={'x': 'Layer', 'y': 'Count'},
            title="Which Layer Classified Each Transaction"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        st.subheader("Confidence Distribution")
        fig = px.histogram(
            results_df,
            x='confidence',
            nbins=20,
            title="Distribution of Confidence Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Results")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            selected_category = st.multiselect(
                "Filter by Category",
                options=results_df['category'].unique()
            )
        with col2:
            min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0)
        
        filtered_df = results_df.copy()
        if selected_category:
            filtered_df = filtered_df[filtered_df['category'].isin(selected_category)]
        filtered_df = filtered_df[filtered_df['confidence'] >= min_conf]
        
        st.dataframe(
            filtered_df[['transaction_id', 'original_description', 'category', 'confidence', 'layer_used', 'reason']],
            use_container_width=True
        )
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            "categorization_results.csv",
            "text/csv"
        )
    else:
        st.info("👆 Upload and classify transactions first")

with tab3:
    st.header("Performance Metrics")
    st.markdown("**Unsupervised System - No Ground Truth Required**")
    
    if st.session_state.results:
        metrics = st.session_state.metrics_tracker.compute_metrics()
        
        # Overall metrics
        st.subheader("📊 Overall Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", metrics['total_transactions'])
            st.metric("Unique Categories", metrics['unique_categories'])
        
        with col2:
            st.metric("Avg Confidence", f"{metrics['avg_confidence']:.2%}")
            st.metric("Median Confidence", f"{metrics['median_confidence']:.2%}")
        
        with col3:
            st.metric("Auto-Label Rate", f"{metrics['auto_label_rate']:.2%}", 
                     help="% with confidence ≥0.75")
            st.metric("Probable Rate", f"{metrics['probable_rate']:.2%}",
                     help="% with confidence 0.50-0.75")
        
        with col4:
            st.metric("Low Confidence", f"{metrics['low_confidence_rate']:.2%}",
                     help="% with confidence <0.50")
            st.metric("Correction Rate", f"{metrics.get('correction_rate', 0.0):.2%}")
        
        # Confidence distribution
        st.subheader("📈 Confidence Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Percentiles**")
            perc_df = pd.DataFrame([
                {'Percentile': k, 'Confidence': f"{v:.2%}"}
                for k, v in metrics['confidence_percentiles'].items()
            ])
            st.dataframe(perc_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Statistics**")
            stats_df = pd.DataFrame([
                {'Metric': 'Min', 'Value': f"{metrics['min_confidence']:.2%}"},
                {'Metric': 'Max', 'Value': f"{metrics['max_confidence']:.2%}"},
                {'Metric': 'Std Dev', 'Value': f"{metrics['std_confidence']:.2%}"}
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Layer performance
        st.subheader("🎯 Layer Performance")
        if 'layer_distribution' in metrics:
            layer_df = pd.DataFrame([
                {
                    'Layer': k, 
                    'Count': v, 
                    'Percentage': f"{metrics['layer_percentages'][k]:.1f}%",
                    'Avg Confidence': f"{metrics['layer_confidence'][k]['avg_confidence']:.2%}"
                }
                for k, v in metrics['layer_distribution'].items()
            ])
            st.dataframe(layer_df, use_container_width=True, hide_index=True)
        
        # Category performance
        st.subheader("📊 Category-Wise Performance")
        cat_summary = st.session_state.metrics_tracker.get_category_summary()
        st.dataframe(cat_summary.style.format({
            'Percentage': '{:.1f}%',
            'Avg Confidence': '{:.2%}',
            'Min Confidence': '{:.2%}',
            'Max Confidence': '{:.2%}',
            'Auto-Label Rate': '{:.1f}%'
        }), use_container_width=True, hide_index=True)
        
        # Gating statistics
        if 'gating_stats' in metrics:
            try:
                st.subheader("⚖️ Gating Network Statistics")
                gating = metrics['gating_stats']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg α (Alpha)", f"{gating.get('avg_alpha', 0.0):.2f}")
                with col2:
                    st.metric("Text Dominant", f"{gating.get('text_dominant_rate', 0.0):.1%}",
                             help="% where α ≥ 0.5 (trusts text more)")
                with col3:
                    st.metric("Behavior Dominant", f"{gating.get('behavior_dominant_rate', 0.0):.1%}",
                             help="% where α < 0.5 (trusts behavior more)")
            except Exception as e:
                st.error(f"⚠️ Error displaying gating statistics: {str(e)}")
        
        # Merchant consistency
        if 'merchant_consistency' in metrics:
            try:
                st.subheader("🏪 Merchant Consistency")
                merchant = metrics['merchant_consistency']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Consistency", 
                             f"{merchant.get('avg_consistency', 0.0):.1%}",
                             help="How consistently same merchant gets same category")
                with col2:
                    st.metric("Merchants Tracked", 
                             merchant.get('merchants_tracked', 0))
            except Exception as e:
                st.error(f"⚠️ Error displaying merchant consistency: {str(e)}")
        
        # Processing stats
        if 'processing_stats' in metrics:
            try:
                st.subheader("⚡ Processing Statistics")
                proc = metrics['processing_stats']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Time per Transaction", 
                             f"{proc.get('avg_time_per_txn', 0.0):.3f}s")
                with col2:
                    st.metric("Total Processing Time", 
                             f"{proc.get('total_processing_time', 0.0):.2f}s")
            except Exception as e:
                st.error(f"⚠️ Error displaying processing statistics: {str(e)}")
        
        # Clustering quality metrics
        if 'clustering_quality' in metrics:
            try:
                st.subheader("🎯 Clustering Quality Metrics")
                st.markdown("*As per IEEE Paper Performance Comparison*")
                
                cq = metrics.get('clustering_quality', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    sil_score = cq.get('silhouette_score', 0.0)
                    st.metric("Silhouette Score", 
                             f"{sil_score:.2f}" if sil_score is not None else "N/A",
                             help="Higher is better (-1 to 1). Our approach: 0.52")
                with col2:
                    db_index = cq.get('davies_bouldin_index', 0.0)
                    st.metric("Davies-Bouldin Index", 
                             f"{db_index:.2f}" if db_index is not None else "N/A",
                             help="Lower is better (0 to ∞). Our approach: 0.72")
                with col3:
                    if 'v_measure' in cq and cq['v_measure'] is not None:
                        st.metric("V-Measure", 
                                 f"{cq['v_measure']:.2f}",
                                 help="Higher is better (0 to 1). Our approach: 0.84")
                    else:
                        st.metric("V-Measure", 
                                 "N/A",
                                 help="Requires ground truth labels")
                
                # Cluster statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Number of Clusters", cq.get('n_clusters', 0))
                with col2:
                    st.metric("Noise Points", cq.get('n_noise_points', 0))
                with col3:
                    noise_ratio = cq.get('noise_ratio', 0.0)
                    st.metric("Noise Ratio", f"{noise_ratio:.1%}")
                with col4:
                    avg_size = cq.get('avg_cluster_size', 0.0)
                    st.metric("Avg Cluster Size", f"{avg_size:.0f}")
            except Exception as e:
                st.error(f"⚠️ Error displaying clustering quality metrics: {str(e)}")
            
            # Comparison with paper
            try:
                st.markdown("**📊 Comparison with IEEE Paper (Table 1)**")
                comparison_df = pd.DataFrame([
                    {
                        'Approach': 'Semantic only',
                        'Silhouette': 0.34,
                        'DB Index': 1.18,
                        'V-measure': 0.65
                    },
                    {
                        'Approach': 'Behavioral only',
                        'Silhouette': 0.41,
                        'DB Index': 0.96,
                        'V-measure': 0.70
                    },
                    {
                        'Approach': 'Fixed 50-50 fusion',
                        'Silhouette': 0.47,
                        'DB Index': 0.84,
                        'V-measure': 0.78
                    },
                    {
                        'Approach': 'Adaptive fusion (Ours)',
                        'Silhouette': 0.52,
                        'DB Index': 0.72,
                        'V-measure': 0.84
                    },
                    {
                        'Approach': 'Your System',
                        'Silhouette': cq.get('silhouette_score', 'N/A'),
                        'DB Index': cq.get('davies_bouldin_index', 'N/A'),
                        'V-measure': cq.get('v_measure', 'N/A')
                    }
                ])
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display comparison table: {str(e)}")
        
        # Export metrics
        if st.button("📥 Export Metrics JSON"):
            st.session_state.metrics_tracker.export_metrics('metrics_export.json')
            st.success("✅ Metrics exported to metrics_export.json")
    else:
        st.info("👆 Classify transactions to see metrics")

with tab4:
    st.header("System Settings")
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Semantic Search Thresholds**")
        st.number_input("Unanimous threshold", 0.0, 1.0, 0.78, 0.01, key='unanimous_thresh')
        st.number_input("Majority threshold", 0.0, 1.0, 0.70, 0.01, key='majority_thresh')
    
    with col2:
        st.markdown("**Classification Thresholds**")
        st.number_input("Auto-label threshold", 0.0, 1.0, 0.75, 0.01, key='auto_label_thresh')
        st.number_input("Request feedback threshold", 0.0, 1.0, 0.50, 0.01, key='feedback_thresh')
    
    st.subheader("Category Management")
    
    categories = [
        'Food & Dining', 'Commute/Transport', 'Shopping', 'Bills & Utilities',
        'Entertainment', 'Healthcare', 'Education', 'Investments',
        'Salary/Income', 'Transfers', 'Subscriptions', 'Others/Uncategorized'
    ]
    
    st.multiselect("Active Categories", categories, default=categories, key='active_categories')
    
    st.subheader("System Info")
    st.info("""
    - **Embedding Model**: intfloat/e5-base-v2 (768-dim)
    - **Clustering**: HDBSCAN
    - **Semantic Search**: FAISS IndexFlatIP
    - **Gating**: MLP (128->32->1)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
if st.session_state.initialized:
    st.sidebar.success("✅ Initialized")
else:
    st.sidebar.warning("⏳ Not initialized")

if st.session_state.transactions is not None:
    st.sidebar.info(f"📊 {len(st.session_state.transactions)} transactions loaded")

