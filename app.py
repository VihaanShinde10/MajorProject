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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload & Classify", "📊 Results", "📈 Metrics", "🔍 Clusters", "⚙️ Settings"])

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
            'DR/CR_Indicator': 'type',
            'Recipient_Name': 'recipient_name',
            'UPI_ID': 'upi_id',
            'Note': 'note',
            'Balance': 'balance'
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
        
        # Create description from available fields if not present
        if 'description' not in df.columns or df['description'].isna().all():
            # Build description from Recipient_Name, UPI_ID, and Note
            description_parts = []
            
            if 'recipient_name' in df.columns:
                description_parts.append(df['recipient_name'].fillna(''))
            if 'upi_id' in df.columns:
                description_parts.append(df['upi_id'].fillna(''))
            if 'note' in df.columns:
                description_parts.append(df['note'].fillna(''))
            
            # Combine parts, filtering out empty strings
            df['description'] = description_parts[0] if description_parts else 'UPI Transaction'
            for part in description_parts[1:]:
                df['description'] = df['description'] + ' ' + part
            
            # Clean up extra spaces
            df['description'] = df['description'].str.strip().str.replace(r'\s+', ' ', regex=True)
            
            st.info("ℹ️ No 'description' column found. Created from Recipient_Name, UPI_ID, and Note fields.")
        
        # Create merchant column from recipient_name if not present
        if 'merchant' not in df.columns:
            if 'recipient_name' in df.columns:
                df['merchant'] = df['recipient_name']
            else:
                df['merchant'] = df['description']
        
        st.session_state.transactions = df
        
        st.success(f"✅ Loaded {len(df)} transactions")
        
        # Show column info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 Columns found: {', '.join(df.columns.tolist())}")
        with col2:
            has_upi = 'recipient_name' in df.columns and 'upi_id' in df.columns
            if has_upi:
                st.success("✅ UPI fields detected (Recipient_Name, UPI_ID, Note)")
            else:
                st.warning("⚠️ UPI fields not found - accuracy may be lower")
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Required columns check
        required_cols = ['date', 'amount', 'type']  # Removed 'description' as we create it
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
                    classified_embeddings = []
                    classified_categories = []
                    classified_metadata = []
                    
                    # Phase 1: Sequential processing with dynamic index building
                    status_text.text("Phase 1: Initializing sequential classification...")
                    
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
                    
                    # Phase 4: Classify each transaction SEQUENTIALLY
                    status_text.text("Phase 4: Classifying transactions sequentially...")
                    
                    # Rebuild index every N transactions for better accuracy
                    rebuild_frequency = 50
                    
                    for idx, row in df.iterrows():
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        
                        history = df[df.index < idx]
                        
                        # Rebuild semantic index periodically with classified transactions
                        if len(classified_embeddings) >= 10 and (idx % rebuild_frequency == 0 or idx == len(df) - 1):
                            status_text.text(f"Updating semantic index... ({len(classified_embeddings)} transactions)")
                            embeddings_array = np.vstack(classified_embeddings)
                            semantic_searcher.build_index(embeddings_array, classified_categories, classified_metadata)
                            st.session_state.semantic_searcher = semantic_searcher
                            
                            # Also update clustering periodically
                            if len(features_list) >= 20:
                                features_df_partial = pd.DataFrame(features_list)
                                cluster_ids = clusterer.fit(features_df_partial, classified_categories)
                                st.session_state.clusterer = clusterer
                        
                        # Layer 0: Rules (highest priority - corpus-based)
                        rule_result = rule_detector.detect(row, history)
                        
                        # Layer 1: Normalization (Enhanced with UPI fields)
                        text = str(row.get('description', ''))
                        recipient_name = row.get('recipient_name', None)
                        upi_id = row.get('upi_id', None)
                        note = row.get('note', None)
                        
                        normalized_text, norm_metadata = normalizer.normalize(
                            text, 
                            recipient_name=recipient_name,
                            upi_id=upi_id,
                            note=note
                        )
                        
                        # DISABLED: Layer 1 canonical match - too aggressive
                        # Let Layer 3 (semantic) handle most merchant matching
                        # Only keep this for VERY high confidence (>0.98)
                        if norm_metadata.get('canonical_match') and norm_metadata.get('canonical_confidence', 0) > 0.98:
                            canonical = norm_metadata['canonical_match']
                            # Only for top brands (netflix, amazon, etc.)
                            top_brands = ['netflix', 'amazon', 'swiggy', 'zomato', 'uber', 'ola', 'spotify']
                            if canonical in top_brands:
                                category = normalizer.category_map.get(canonical, 'Others/Uncategorized')
                                
                                result = {
                                    'transaction_id': idx,
                                    'original_description': text,
                                    'normalized_text': normalized_text,
                                    'recipient_name': recipient_name if recipient_name and not pd.isna(recipient_name) else '',
                                    'upi_id': upi_id if upi_id and not pd.isna(upi_id) else '',
                                    'note': note if note and not pd.isna(note) else '',
                                    'amount': row['amount'],
                                    'category': category,
                                    'confidence': norm_metadata['canonical_confidence'],
                                    'layer_used': 'L1: Canonical Match',
                                    'reason': f"Exact match to top brand: {canonical}",
                                    'should_prompt': False,
                                    'alpha': None
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
                        
                        # Layer 2 & 3: Embeddings + Semantic Search (Enhanced)
                        embedding = embedder.embed(
                            normalized_text, 
                            transaction_mode=row.get('type', ''),
                            recipient_name=recipient_name,
                            upi_id=upi_id,
                            note=note,
                            amount=row['amount']
                        )
                        
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
                        
                        # Layer 8: Zero-shot (VERY RESTRICTED - only for complete failures)
                        zeroshot_result = (None, 0.0, {})
                        # STRICTER conditions: Only use L8 if ALL layers failed AND user explicitly enabled it
                        use_layer8 = (
                            use_zeroshot and 
                            zeroshot_classifier and
                            (not rule_result[0]) and  # L0 failed
                            (not semantic_result[0] or semantic_result[1] < 0.40) and  # L3 weak/failed
                            (not behavioral_result[0] or behavioral_result[1] < 0.40)  # L5 weak/failed
                        )
                        
                        if use_layer8:
                            # Last resort: zero-shot classification
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
                            'recipient_name': recipient_name if recipient_name and not pd.isna(recipient_name) else '',
                            'upi_id': upi_id if upi_id and not pd.isna(upi_id) else '',
                            'note': note if note and not pd.isna(note) else '',
                            'amount': row['amount'],
                            'category': classification.category,
                            'confidence': classification.confidence,
                            'layer_used': classification.layer_used,
                            'reason': classification.reason,
                            'should_prompt': classification.should_prompt,
                            'alpha': alpha
                        }
                        
                        results.append(result)
                        
                        # Store for future index building (sequential learning)
                        if classification.confidence >= 0.60:  # Only store confident predictions
                            classified_embeddings.append(embedding)
                            classified_categories.append(classification.category)
                            classified_metadata.append({
                                'idx': idx,
                                'description': text,
                                'confidence': classification.confidence
                            })
                        
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
        
        # Display with new UPI fields
        display_columns = ['transaction_id', 'original_description', 'recipient_name', 'upi_id', 
                          'note', 'amount', 'category', 'confidence', 'layer_used', 'reason']
        # Only show columns that exist
        available_cols = [col for col in display_columns if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols],
            use_container_width=True,
            column_config={
                'recipient_name': st.column_config.TextColumn('Recipient', width='medium'),
                'upi_id': st.column_config.TextColumn('UPI ID', width='medium'),
                'note': st.column_config.TextColumn('Note', width='small'),
                'amount': st.column_config.NumberColumn('Amount', format='₹%.2f'),
                'confidence': st.column_config.ProgressColumn('Confidence', format='%.2f', min_value=0, max_value=1)
            }
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
    st.header("🔍 Cluster Analysis")
    
    if st.session_state.clusterer and st.session_state.clusterer.clusterer:
        clusterer = st.session_state.clusterer
        
        # Cluster Overview
        st.subheader("Cluster Overview")
        
        labels = clusterer.clusterer.labels_
        unique_clusters = set(labels)
        unique_clusters.discard(-1)  # Remove noise
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clusters", len(unique_clusters))
        with col2:
            noise_count = sum(1 for l in labels if l == -1)
            st.metric("Noise Points", noise_count)
        with col3:
            if len(labels) > 0:
                noise_ratio = noise_count / len(labels) * 100
                st.metric("Noise Ratio", f"{noise_ratio:.1f}%")
        
        # Cluster Distribution
        st.subheader("Cluster Size Distribution")
        
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            count = sum(1 for l in labels if l == cluster_id)
            cluster_sizes[cluster_id] = count
        
        if cluster_sizes:
            # Bar chart
            cluster_df = pd.DataFrame([
                {'Cluster ID': f'Cluster {cid}', 'Size': size, 'Category': clusterer.cluster_labels.get(cid, 'Unknown')}
                for cid, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            ])
            
            fig = px.bar(
                cluster_df,
                x='Cluster ID',
                y='Size',
                color='Category',
                title='Transactions per Cluster',
                labels={'Size': 'Number of Transactions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Details Table
        st.subheader("Cluster Details")
        
        cluster_details = []
        for cluster_id in sorted(unique_clusters):
            count = cluster_sizes.get(cluster_id, 0)
            category = clusterer.cluster_labels.get(cluster_id, 'Unknown')
            
            # Calculate cluster density (cohesion)
            cluster_mask = labels == cluster_id
            if clusterer.feature_vectors is not None:
                cluster_features = clusterer.feature_vectors[cluster_mask]
                if len(cluster_features) > 1:
                    from sklearn.metrics.pairwise import euclidean_distances
                    distances = euclidean_distances(cluster_features)
                    avg_distance = np.mean(distances)
                    cohesion = 1 / (1 + avg_distance)  # Convert distance to cohesion score
                else:
                    cohesion = 1.0
            else:
                cohesion = 0.0
            
            cluster_details.append({
                'Cluster ID': cluster_id,
                'Category': category,
                'Size': count,
                'Cohesion Score': f"{cohesion:.3f}"
            })
        
        if cluster_details:
            details_df = pd.DataFrame(cluster_details)
            st.dataframe(details_df, use_container_width=True)
        
        # Cluster Quality Metrics
        st.subheader("Cluster Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Silhouette Score**")
            if clusterer.feature_vectors is not None and len(unique_clusters) > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    # Only compute for non-noise points
                    non_noise_mask = labels != -1
                    if sum(non_noise_mask) > 1:
                        score = silhouette_score(
                            clusterer.feature_vectors[non_noise_mask],
                            labels[non_noise_mask]
                        )
                        st.metric("Silhouette Score", f"{score:.3f}")
                        st.caption("Range: [-1, 1]. Higher is better. >0.5 is good.")
                    else:
                        st.info("Not enough non-noise points")
                except Exception as e:
                    st.warning(f"Could not compute: {str(e)}")
            else:
                st.info("Need at least 2 clusters")
        
        with col2:
            st.markdown("**Davies-Bouldin Index**")
            if clusterer.feature_vectors is not None and len(unique_clusters) > 1:
                try:
                    from sklearn.metrics import davies_bouldin_score
                    non_noise_mask = labels != -1
                    if sum(non_noise_mask) > 1:
                        score = davies_bouldin_score(
                            clusterer.feature_vectors[non_noise_mask],
                            labels[non_noise_mask]
                        )
                        st.metric("Davies-Bouldin Index", f"{score:.3f}")
                        st.caption("Range: [0, ∞]. Lower is better. <1 is good.")
                    else:
                        st.info("Not enough non-noise points")
                except Exception as e:
                    st.warning(f"Could not compute: {str(e)}")
            else:
                st.info("Need at least 2 clusters")
        
        # 2D Visualization (using PCA)
        st.subheader("Cluster Visualization (2D PCA)")
        
        if clusterer.feature_vectors is not None and len(clusterer.feature_vectors) > 0:
            try:
                from sklearn.decomposition import PCA
                
                # Apply PCA
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(clusterer.feature_vectors)
                
                # Create DataFrame for plotting
                viz_df = pd.DataFrame({
                    'PC1': features_2d[:, 0],
                    'PC2': features_2d[:, 1],
                    'Cluster': ['Noise' if l == -1 else f'Cluster {l}' for l in labels],
                    'Category': [clusterer.cluster_labels.get(l, 'Noise') for l in labels]
                })
                
                fig = px.scatter(
                    viz_df,
                    x='PC1',
                    y='PC2',
                    color='Category',
                    symbol='Cluster',
                    title='Transaction Clusters (PCA Projection)',
                    labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"PCA Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
                
            except Exception as e:
                st.error(f"Could not create visualization: {str(e)}")
        
    else:
        st.info("👆 Classify transactions first to see cluster analysis")

with tab5:
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

