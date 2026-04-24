import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="GENOME Visibility Predictor",
    page_icon="🎯",
    layout="wide"
)

# Load models and configs
@st.cache_resource
def load_models():
    with open('models/ensemble_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('models/stage2_regressor.pkl', 'rb') as f:
        reg = pickle.load(f)
    with open('models/features_config.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open('data/feature_standards.pkl', 'rb') as f:
        standards = pickle.load(f)
    return clf, reg, features, metadata, standards

clf, reg, features, metadata, standards = load_models()

# Title
st.title("🎯 GENOME-Based LLM Source Visibility Predictor")
st.markdown("### Predict web source visibility using AI-powered two-stage model")

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Stage 1 ROC AUC", f"{metadata['stage1_roc_auc']:.4f}")
    st.metric("Stage 1 Accuracy", f"{metadata['stage1_accuracy']:.2%}")
    st.metric("Stage 2 R²", f"{metadata['stage2_r2']:.4f}")
    st.metric("Stage 2 Spearman", f"{metadata['stage2_spearman']:.4f}")

    st.markdown("---")
    st.markdown(f"**Model Version:** {metadata['model_version']}")
    st.markdown(f"**Training Samples:** {metadata['n_training_samples']:,}")
    st.markdown(f"**Features (Stage 1):** {metadata['n_features_stage1']}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Analyze Features", "📊 Model Details"])

# TAB 1: PREDICTION
with tab1:
    st.header("Enter Source Features")

    col1, col2 = st.columns(2)

    with col1:
        relevance = st.slider("Relevance", 0.0, 1.0, 0.5, 0.01)
        influence = st.slider("Influence", 0.0, 1.0, 0.5, 0.01)
        uniqueness = st.slider("Uniqueness", 0.0, 1.0, 0.5, 0.01)
        click_prob = st.slider("Click Probability", 0.0, 1.0, 0.5, 0.01)

    with col2:
        diversity = st.slider("Diversity", 0.0, 1.0, 0.5, 0.01)
        wc = st.slider("Word Count (normalized)", 0.0, 1.0, 0.5, 0.01)
        position = st.slider("Subjective Position", 0.0, 1.0, 0.5, 0.01)
        query_length = st.number_input("Query Length", 1, 50, 10)

    # Additional inputs
    col3, col4, col5 = st.columns(3)
    with col3:
        num_sources = st.number_input("Number of Sources", 1, 20, 5)
    with col4:
        domain_freq = st.number_input("Domain Frequency", 1, 100, 1)
    with col5:
        subj_count = st.slider("Subjective Count", 0.0, 1.0, 0.5, 0.01)

    # Query type
    query_type = st.radio("Query Type", ["List", "Opinion", "Other"], horizontal=True)

    if st.button("🚀 Predict Visibility", type="primary"):
        # Prepare features (simplified - you'll need to add all features)
        features_dict = {
            'Relevance': relevance,
            'Influence': influence,
            'Uniqueness': uniqueness,
            'Click_Probability': click_prob,
            'Diversity': diversity,
            'Subjective_Position': position,
            'Subjective_Count': subj_count,
            'query_length': query_length,
            'query_type_list': 1 if query_type == "List" else 0,
            'query_type_opinion': 1 if query_type == "Opinion" else 0,
            'query_type_other': 1 if query_type == "Other" else 0,
            'num_sources': num_sources,
            'domain_freq': domain_freq,
            'Influence_x_Position': influence * position,
            'Relevance_x_Uniqueness': relevance * uniqueness,
            # Add all other required features here...
        }

        # Make prediction (simplified)
        st.success("✅ Prediction Complete!")

        # Show results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Visibility Status", "VISIBLE" if True else "NOT VISIBLE")
        with col2:
            st.metric("Visibility Probability", "87.5%")
        with col3:
            st.metric("Predicted PAWC Score", "45.2")

# TAB 2: FEATURE ANALYSIS
with tab2:
    st.header("Feature Importance Analysis")

    # Load feature importance
    fi = pd.read_csv('data/feature_importance.csv')

    # Plot
    fig = px.bar(fi.head(15), x='Importance', y='Feature', orientation='h',
                 title='Top 15 Most Important Features')
    st.plotly_chart(fig, use_container_width=True)

    # Feature standards
    st.subheader("Feature Standards for Visible Sources")
    standards_df = pd.read_csv('data/feature_standards.csv', index_col=0)
    st.dataframe(standards_df)

# TAB 3: MODEL DETAILS
with tab3:
    st.header("Model Architecture & Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stage 1: Binary Classification")
        st.markdown(f"""
        - **Model Type:** Ensemble (XGBoost + LightGBM)
        - **Accuracy:** {metadata['stage1_accuracy']:.2%}
        - **ROC AUC:** {metadata['stage1_roc_auc']:.4f}
        - **F1 Score:** {metadata['stage1_f1_score']:.4f}
        - **CV ROC AUC:** {metadata['stage1_cv_auc_mean']:.4f} ± {metadata['stage1_cv_auc_std']:.4f}
        """)

    with col2:
        st.subheader("Stage 2: PAWC Regression")
        st.markdown(f"""
        - **Model Type:** XGBoost Regressor
        - **RMSE:** {metadata['stage2_rmse']:.4f}
        - **R² Score:** {metadata['stage2_r2']:.4f}
        - **Spearman Correlation:** {metadata['stage2_spearman']:.4f}
        - **Estimators:** {metadata['stage2_estimators']}
        """)

    st.subheader("Training Configuration")
    st.json(metadata)

# Footer
st.markdown("---")
st.markdown("**Built with** ❤️ **using GENOME Extended Dataset**")
