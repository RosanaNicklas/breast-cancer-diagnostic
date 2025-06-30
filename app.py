import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.utils.validation import check_is_fitted

# Set page config
st.set_page_config(
    page_title="Breast Cancer Diagnostic Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data with proper feature name handling
@st.cache_resource
def load_model():
    model = joblib.load('breast_cancer_model.pkl')
    check_is_fitted(model)
    return model

@st.cache_data
def load_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=[str(f) for f in data.feature_names])
    df['diagnosis'] = data.target
    return df, [str(f) for f in data.feature_names]

try:
    model = load_model()
    df, feature_names = load_dataset()
    
    # Ensure feature names match model expectations
    if hasattr(model, 'feature_names_in_'):
        feature_names = [str(f) for f in model.feature_names_in_]
except Exception as e:
    st.error(f"Error loading model or data: {str(e)}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .diagnosis-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    .malignant { background-color: #ffdddd; color: #d32f2f; }
    .benign { background-color: #ddffdd; color: #388e3c; }
    .feature-importance { background-color: white; border-radius: 10px; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# App layout
st.title("Breast Cancer Diagnostic Assistant")
st.markdown("""
This tool assists medical professionals in evaluating breast mass characteristics to predict malignancy risk.
All predictions should be interpreted in clinical context.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Patient Data Input")
    st.markdown("Enter the tumor characteristics from the FNA biopsy:")
    
    # Create input widgets for all required features
    input_features = {}
    with st.expander("Size Characteristics"):
        input_features['mean radius'] = st.slider(
            "Mean Radius (mean of distances from center to points on the perimeter)", 
            min_value=float(df['mean radius'].min()), 
            max_value=float(df['mean radius'].max()), 
            value=float(df['mean radius'].mean())
        )
        input_features['mean perimeter'] = st.slider(
            "Mean Perimeter", 
            min_value=float(df['mean perimeter'].min()), 
            max_value=float(df['mean perimeter'].max()), 
            value=float(df['mean perimeter'].mean())
        )

    with st.expander("Texture Characteristics"):
        input_features['mean texture'] = st.slider(
            "Mean Texture (standard deviation of gray-scale values)", 
            min_value=float(df['mean texture'].min()), 
            max_value=float(df['mean texture'].max()), 
            value=float(df['mean texture'].mean())
        )

    with st.expander("Other Key Features"):
        input_features['mean concavity'] = st.slider(
            "Mean Concavity (severity of concave portions of the contour)", 
            min_value=float(df['mean concavity'].min()), 
            max_value=float(df['mean concavity'].max()), 
            value=float(df['mean concavity'].mean())
        )
        input_features['mean symmetry'] = st.slider(
            "Mean Symmetry", 
            min_value=float(df['mean symmetry'].min()), 
            max_value=float(df['mean symmetry'].max()), 
            value=float(df['mean symmetry'].mean())
        )

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Run Diagnostic Analysis", type="primary"):
        try:
            # Create input dataframe with all expected features
            input_data = pd.DataFrame({
                feature: [input_features.get(feature, df[feature].mean())] 
                for feature in feature_names
            })
            
            # Ensure column names match exactly what the model expects
            input_data.columns = [str(col) for col in input_data.columns]
            
            # Get prediction and probabilities
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Display diagnosis
            if prediction == 0:
                st.markdown(f"""
                <div class="diagnosis-box malignant">
                    <h3>Diagnostic Result: MALIGNANT</h3>
                    <p>Probability: {probabilities[0]*100:.1f}%</p>
                    <p>Clinical recommendation: Further investigation required</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="diagnosis-box benign">
                    <h3>Diagnostic Result: BENIGN</h3>
                    <p>Probability: {probabilities[1]*100:.1f}%</p>
                    <p>Clinical recommendation: Routine follow-up suggested</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show probability distribution
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(['Benign', 'Malignant'], probabilities, color=['#4CAF50', '#F44336'])
            ax.set_xlim(0, 1)
            ax.set_title('Prediction Confidence')
            ax.set_xlabel('Probability')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

with col2:
    st.header("Clinical Context")
    
    # Feature importance visualization
    st.subheader("Key Predictive Factors")
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_features = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
            plt.title('Top 10 Most Important Features')
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model type")
    except Exception as e:
        st.error(f"Could not display feature importance: {str(e)}")
    
    # Data distribution comparison
    st.subheader("Patient Values vs. Typical Ranges")
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(df[df['diagnosis'] == 0]['mean radius'], label='Malignant', ax=ax)
        sns.kdeplot(df[df['diagnosis'] == 1]['mean radius'], label='Benign', ax=ax)
        ax.axvline(input_features['mean radius'], color='red', linestyle='--', label='Current Patient')
        ax.set_title('Mean Radius Distribution Comparison')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not create distribution plot: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This tool provides statistical predictions based on machine learning models. 
It is not a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider.
""")