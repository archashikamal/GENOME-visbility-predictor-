import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import urlparse
import re

# Page config
st.set_page_config(
    page_title="GENOME Visibility Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .feature-comparison {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load models and configs
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base_dir, 'models', 'ensemble_classifier.pkl'), 'rb') as f:
            clf = pickle.load(f)
        with open(os.path.join(base_dir, 'models', 'stage2_regressor.pkl'), 'rb') as f:
            reg = pickle.load(f)
        with open(os.path.join(base_dir, 'models', 'features_config.pkl'), 'rb') as f:
            features = pickle.load(f)
        with open(os.path.join(base_dir, 'models', 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        with open(os.path.join(base_dir, 'data', 'feature_standards.pkl'), 'rb') as f:
            standards = pickle.load(f)
        return clf, reg, features, metadata, standards
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# Extract features from URL
def extract_features_from_url(url, query):
    """
    Extract features from a URL and query
    This is a simplified version - in production, you'd use more sophisticated methods
    """
    features = {}
    
    try:
        # Fetch the webpage
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        text = soup.get_text(separator=' ', strip=True)
        word_count = len(text.split())
        
        # Normalize word count (assume max reasonable is 5000 words)
        features['WC'] = min(word_count / 5000, 1.0)
        features['WC_rel'] = features['WC']  # Simplified
        
        # Extract domain info
        domain = urlparse(url).netloc
        features['domain_freq'] = 1  # Default, would need historical data
        
        # Query analysis
        query_words = query.lower().split()
        features['query_length'] = len(query_words)
        
        # Detect query type (simplified)
        list_keywords = ['list', 'top', 'best', 'names', 'examples']
        opinion_keywords = ['should', 'why', 'how', 'opinion', 'think']
        
        is_list = any(word in query.lower() for word in list_keywords)
        is_opinion = any(word in query.lower() for word in opinion_keywords)
        
        features['query_type_list'] = 1 if is_list else 0
        features['query_type_opinion'] = 1 if is_opinion else 0
        features['query_type_other'] = 1 if not (is_list or is_opinion) else 0
        
        # Calculate relevance (simplified - keyword matching)
        query_terms = set(query.lower().split())
        text_lower = text.lower()
        matching_terms = sum(1 for term in query_terms if term in text_lower)
        features['Relevance'] = min(matching_terms / max(len(query_terms), 1), 1.0)
        
        # Influence (based on domain factors - simplified)
        # In production, use domain authority APIs
        domain_score = 0.5  # Default medium influence
        if any(tld in domain for tld in ['.edu', '.gov']):
            domain_score = 0.9
        elif any(tld in domain for tld in ['.org', '.com']):
            domain_score = 0.6
        features['Influence'] = domain_score
        
        # Uniqueness (based on content diversity - simplified)
        unique_words = len(set(text.lower().split()))
        features['Uniqueness'] = min(unique_words / max(word_count, 1), 1.0)
        
        # Click probability (based on title and meta description)
        title = soup.find('title')
        title_text = title.get_text() if title else ""
        features['Click_Probability'] = 0.7 if len(title_text) > 0 else 0.3
        
        # Diversity (content variety - simplified)
        has_images = len(soup.find_all('img')) > 0
        has_headings = len(soup.find_all(['h1', 'h2', 'h3'])) > 0
        has_lists = len(soup.find_all(['ul', 'ol'])) > 0
        diversity_score = sum([has_images, has_headings, has_lists]) / 3
        features['Diversity'] = diversity_score
        
        # Position metrics (defaults)
        features['Subjective_Position'] = 0.5
        features['Subjective_Count'] = 0.5
        
        # Other required features
        features['num_sources'] = 5  # Default
        features['Influence_x_Position'] = features['Influence'] * features['Subjective_Position']
        features['Relevance_x_Uniqueness'] = features['Relevance'] * features['Uniqueness']
        
        # Engineered features (using same formulas as training)
        features['Content_Depth'] = features['WC'] * features['Relevance']
        features['Authority_Score'] = features['Influence'] * features['Uniqueness']
        features['Engagement_Potential'] = features['Click_Probability'] * features['Diversity']
        features['Uniqueness_x_Relevance'] = features['Uniqueness'] * features['Relevance']
        
        # Quality scores
        features['Quality_Score'] = (
            features['Relevance'] * 0.4 +
            features['Influence'] * 0.3 +
            features['Uniqueness'] * 0.3
        )
        
        # More engineered features (simplified - many require query-level data)
        features['Relative_Quality'] = 1.0
        features['Influence_Advantage'] = 0.0
        features['Quality_Advantage'] = 0.0
        features['Position_Score'] = features['Influence'] / (features['Subjective_Position'] + 1)
        features['Weighted_Relevance'] = features['Relevance'] / (features['Subjective_Position'] + 1)
        features['Domain_Influence'] = features['domain_freq'] * features['Influence']
        features['Domain_Consistency'] = features['domain_freq'] / (features['num_sources'] + 1)
        features['Domain_Has_History'] = 1 if features['domain_freq'] > 1 else 0
        
        # Query interactions
        query_type_multiplier = (
            features['query_type_list'] * 1.2 + 
            features['query_type_opinion'] * 1.1 + 
            features['query_type_other'] * 1.0
        )
        features['Relevance_x_QueryType'] = features['Relevance'] * query_type_multiplier
        
        features['Overall_Score'] = (
            features['Relevance'] * 0.25 +
            features['Influence'] * 0.25 +
            features['Uniqueness'] * 0.20 +
            features['Click_Probability'] * 0.15 +
            features['Diversity'] * 0.15
        )
        
        # Rank features (defaults since we don't have multiple sources)
        features['Influence_rank'] = 0.5
        features['Relevance_rank'] = 0.5
        features['Uniqueness_rank'] = 0.5
        features['Click_Prob_rank'] = 0.5
        features['Relevance_percentile'] = 0.5
        features['Influence_percentile'] = 0.5
        features['Uniqueness_percentile'] = 0.5
        features['Quality_Score_percentile'] = 0.5
        
        # Z-scores (defaults)
        features['Relevance_zscore'] = 0.0
        features['Influence_zscore'] = 0.0
        features['Uniqueness_zscore'] = 0.0
        
        # Binary features
        features['High_Quality'] = 1 if features['Quality_Score'] > 0.6 else 0
        features['High_Influence'] = 1 if features['Influence'] > 0.6 else 0
        features['Top_Position'] = 1 if features['Subjective_Position'] <= 0.25 else 0
        features['Quality_Position_Match'] = features['High_Quality'] * features['Top_Position']
        features['Influence_Relevance_Match'] = features['High_Influence'] * (1 if features['Relevance'] > 0.7 else 0)
        
        # Competition metrics (defaults)
        features['Query_Competition'] = 0.2
        features['Adjusted_Quality'] = features['Quality_Score'] / (features['Query_Competition'] + 0.1)
        
        return features, text[:500]  # Return features and text preview
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None

# Predict visibility
def predict_visibility(features, clf, reg, stage1_features, stage2_features, threshold):
    """Make visibility prediction"""
    try:
        # Stage 1: Binary classification
        X1 = np.array([features[f] for f in stage1_features]).reshape(1, -1)
        is_visible = bool(clf.predict(X1)[0])
        visibility_prob = float(clf.predict_proba(X1)[0][1])
        
        # Stage 2: PAWC prediction (if visible)
        pawc_score = None
        if is_visible:
            X2 = np.array([features[f] for f in stage2_features]).reshape(1, -1)
            log_pawc = reg.predict(X2)[0]
            pawc_score = float(np.expm1(log_pawc))
            # Fix: Clamp to valid range (0-100)
            pawc_score = max(0, min(pawc_score, 100))
        
        return is_visible, visibility_prob, pawc_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# Compare with standards
def compare_with_standards(features, standards):
    """Compare extracted features with visibility standards"""
    comparison = {}
    gaps = []
    
    core_features = ['Relevance', 'Influence', 'Uniqueness', 'Click_Probability', 'Diversity', 'WC']
    
    for feature in core_features:
        if feature in features and feature in standards:
            actual = features[feature]
            target = standards[feature]['75th']
            mean_val = standards[feature]['mean']
            
            gap_pct = ((target - actual) / target * 100) if target > 0 else 0
            
            comparison[feature] = {
                'actual': actual,
                'target': target,
                'mean': mean_val,
                'gap': gap_pct,
                'status': '✅' if actual >= target * 0.9 else '⚠️' if actual >= mean_val else '❌'
            }
            
            if actual < target * 0.9:
                gaps.append({
                    'feature': feature,
                    'actual': actual,
                    'target': target,
                    'gap': gap_pct
                })
    
    gaps.sort(key=lambda x: x['gap'], reverse=True)
    return comparison, gaps

# Main app
def main():
    clf, reg, features_config, metadata, standards = load_models()
    
    if clf is None:
        st.error("Failed to load models. Please check if model files exist.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🎯 GENOME Visibility Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze your website's visibility potential based on learned standards")
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Stage 1 ROC AUC", f"{metadata['stage1_roc_auc']:.4f}")
        st.metric("Accuracy", f"{metadata['stage1_accuracy']:.1%}")
        st.metric("Stage 2 Spearman", f"{metadata['stage2_spearman']:.4f}")
        
        st.markdown("---")
        st.subheader("🎓 Visibility Standards")
        st.markdown(f"""
        Based on analysis of **{metadata['n_training_samples']:,}** sources:
        - **{metadata['visible_ratio']:.1%}** achieved high visibility
        - **{metadata['n_features_stage1']}** features analyzed
        - **Content Depth** is #1 predictor
        """)
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input(
            "🔗 Website URL",
            placeholder="https://www.example.com/article",
            help="Enter the full URL of the webpage you want to analyze"
        )
    
    with col2:
        query = st.text_input(
            "🔍 Search Query",
            placeholder="best skincare products",
            help="Enter the search query this page should rank for"
        )
    
    analyze_button = st.button("🚀 Analyze Visibility", type="primary", use_container_width=True)
    
    if analyze_button and url and query:
        with st.spinner("🔄 Analyzing webpage and extracting features..."):
            # Extract features
            extracted_features, text_preview = extract_features_from_url(url, query)
            
            if extracted_features is None:
                st.error("Could not extract features from URL. Please check the URL and try again.")
                return
            
            # Make prediction
            is_visible, visibility_prob, pawc_score = predict_visibility(
                extracted_features,
                clf,
                reg,
                features_config['stage1_features'],
                features_config['stage2_features'],
                metadata['visibility_threshold']
            )
            
            # Compare with standards
            comparison, gaps = compare_with_standards(extracted_features, standards)
        
        # Display results
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if is_visible else "🔴"
            status_text = "VISIBLE" if is_visible else "NOT VISIBLE"
            st.markdown(f"### {status_color} {status_text}")
            st.caption(f"Threshold: {metadata['visibility_threshold']:.2f}")
        
        with col2:
            prob_color = "🟢" if visibility_prob > 0.7 else "🟡" if visibility_prob > 0.4 else "🔴"
            st.metric("Visibility Probability", f"{visibility_prob:.1%}")
            st.caption(f"{prob_color} Confidence level")
        
        with col3:
            if pawc_score is not None:
                # Categorize visibility level
                if pawc_score < 2.04:
                    level = "🔴 Very Low"
                elif pawc_score < 10:
                    level = "🟡 Moderate"
                else:
                    level = "🟢 Good"
                
                st.metric("Predicted PAWC Score", f"{pawc_score:.2f}")
                st.caption(f"{level} visibility level")
                
                if pawc_score < 2.04:
                    st.caption("⚠️ Below threshold - needs improvement")
            else:
                st.metric("PAWC Score", "N/A")
                st.caption("Not classified as visible")
        
        st.info("""
        ⚠️ **About PAWC Scores**: 
        - Scores are estimates based on extracted features
        - Stage 2 model has R² = 0.27 (moderate accuracy)
        - Use for relative comparison rather than absolute values
        - Threshold: 2.04 (below this = low visibility)
        """)
        
        # Feature comparison
        st.markdown("---")
        st.header("🔍 Feature Analysis vs. Visibility Standards")
        
        # Create comparison chart
        if comparison:
            comparison_df = pd.DataFrame(comparison).T
            comparison_df = comparison_df.reset_index()
            comparison_df.columns = ['Feature', 'actual', 'target', 'mean', 'gap', 'status']
            
            # Bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Your Page',
                x=comparison_df['Feature'],
                y=comparison_df['actual'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Visibility Standard (75th %ile)',
                x=comparison_df['Feature'],
                y=comparison_df['target'],
                marker_color='lightcoral'
            ))
            
            fig.add_trace(go.Scatter(
                name='Average (Visible Sources)',
                x=comparison_df['Feature'],
                y=comparison_df['mean'],
                mode='markers',
                marker=dict(size=12, symbol='diamond', color='gold')
            ))
            
            fig.update_layout(
                barmode='group',
                title="Feature Comparison: Your Page vs. Visibility Standards",
                xaxis_title="Feature",
                yaxis_title="Score (0-1)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.subheader("📋 Detailed Comparison")
            
            for _, row in comparison_df.iterrows():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['status']} {row['Feature']}**")
                
                with col2:
                    st.metric("Your Score", f"{row['actual']:.3f}")
                
                with col3:
                    st.metric("Target", f"{row['target']:.3f}")
                
                with col4:
                    gap_color = "🟢" if row['gap'] <= 10 else "🟡" if row['gap'] <= 30 else "🔴"
                    st.metric("Gap", f"{gap_color} {abs(row['gap']):.0f}%")
        
        # Improvement recommendations
        if gaps:
            st.markdown("---")
            st.header("💡 Improvement Recommendations")
            
            st.markdown(f"**Top {min(3, len(gaps))} areas for improvement:**")
            
            for i, gap in enumerate(gaps[:3], 1):
                with st.expander(f"{i}. Improve {gap['feature']} (Gap: {gap['gap']:.0f}%)"):
                    st.markdown(f"""
                    - **Current Score:** {gap['actual']:.3f}
                    - **Target Score:** {gap['target']:.3f}
                    - **Gap:** {gap['gap']:.0f}%
                    
                    **Recommendations:**
                    """)
                    
                    if gap['feature'] == 'Relevance':
                        st.markdown("""
                        - Include more query keywords in your content
                        - Add related terms and synonyms
                        - Ensure content directly answers the query
                        """)
                    elif gap['feature'] == 'Influence':
                        st.markdown("""
                        - Build domain authority with quality backlinks
                        - Publish consistently high-quality content
                        - Get featured on authoritative sites
                        """)
                    elif gap['feature'] == 'Uniqueness':
                        st.markdown("""
                        - Add original insights and perspectives
                        - Use diverse vocabulary
                        - Provide unique data or examples
                        """)
                    elif gap['feature'] == 'WC':
                        st.markdown("""
                        - Expand content with more details
                        - Add examples and case studies
                        - Include comprehensive explanations
                        """)
                    elif gap['feature'] == 'Click_Probability':
                        st.markdown("""
                        - Write compelling title tags
                        - Craft engaging meta descriptions
                        - Use power words and numbers
                        """)
                    elif gap['feature'] == 'Diversity':
                        st.markdown("""
                        - Add images and videos
                        - Include lists and tables
                        - Use multiple content formats
                        """)
        
        # Content preview
        with st.expander("📄 Content Preview"):
            st.text(text_preview)
        
        # Extracted features (for debugging)
        with st.expander("🔧 Extracted Features (Technical)"):
            features_df = pd.DataFrame([extracted_features]).T
            features_df.columns = ['Value']
            st.dataframe(features_df)
    
    elif analyze_button:
        st.warning("⚠️ Please enter both URL and search query to analyze.")

if __name__ == "__main__":
    main()
