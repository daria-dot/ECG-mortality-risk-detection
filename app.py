"""
ECG Mortality Risk Detection - Streamlit Application

This app provides a user-friendly interface for predicting mortality risk
from ECG data using a trained deep learning model.
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the project modules
import model_architecture
import data_utils
import model_utils

# Constants
MODEL_PATH = 'data/ecg_survival_model.h5'
N_INTERVALS = 120
MAX_TIME_YEARS = 10

# Page configuration
st.set_page_config(
    page_title="ECG Mortality Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
        return None
    
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'survival_loss_with_mask': model_architecture.survival_loss_with_mask}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_ecg_input(ecg_data):
    """
    Process ECG data for model input
    Expected input shape: (4096, 8) for single ECG
    """
    # Ensure correct shape
    if ecg_data.shape != (4096, 8):
        st.error(f"Invalid ECG shape. Expected (4096, 8), got {ecg_data.shape}")
        return None
    
    # Add batch dimension
    ecg_batch = np.expand_dims(ecg_data, axis=0)
    return ecg_batch.astype(np.float32)

def predict_survival(model, ecg_data):
    """
    Make survival predictions using the model
    Returns survival probabilities for each time interval
    """
    # Get predictions
    predictions = model.predict(ecg_data, verbose=0)
    
    # Calculate cumulative survival probabilities
    cumulative_survival = np.cumprod(predictions[0])
    
    return predictions[0], cumulative_survival

def calculate_risk_score(cumulative_survival):
    """
    Calculate risk scores at different time points
    """
    # Define key time points (in months)
    time_points = {
        '6 months': 6,
        '1 year': 12,
        '2 years': 24,
        '5 years': 60,
        '10 years': 119  # Last interval
    }
    
    risk_scores = {}
    for label, month in time_points.items():
        if month < len(cumulative_survival):
            survival_prob = cumulative_survival[month]
            risk_scores[label] = {
                'survival': survival_prob,
                'risk': 1 - survival_prob
            }
    
    return risk_scores

def plot_survival_curve(cumulative_survival):
    """
    Create an interactive survival probability curve
    """
    months = np.arange(len(cumulative_survival))
    years = months / 12
    
    fig = go.Figure()
    
    # Add survival probability line
    fig.add_trace(go.Scatter(
        x=years,
        y=cumulative_survival * 100,
        mode='lines',
        name='Survival Probability',
        line=dict(color='blue', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,100,200,0.2)'
    ))
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, 
                  annotation_text="High Risk", annotation_position="right")
    fig.add_hrect(y0=30, y1=70, fillcolor="yellow", opacity=0.1,
                  annotation_text="Medium Risk", annotation_position="right")
    fig.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.1,
                  annotation_text="Low Risk", annotation_position="right")
    
    # Update layout
    fig.update_layout(
        title="Predicted Survival Probability Over Time",
        xaxis_title="Time (Years)",
        yaxis_title="Survival Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def display_risk_assessment(risk_scores):
    """
    Display risk assessment in a user-friendly format
    """
    st.subheader("📊 Risk Assessment")
    
    cols = st.columns(len(risk_scores))
    
    for idx, (time_point, scores) in enumerate(risk_scores.items()):
        with cols[idx]:
            survival_pct = scores['survival'] * 100
            risk_pct = scores['risk'] * 100
            
            # Determine risk level
            if risk_pct < 10:
                risk_class = "risk-low"
                risk_label = "Low"
            elif risk_pct < 30:
                risk_class = "risk-medium"
                risk_label = "Medium"
            else:
                risk_class = "risk-high"
                risk_label = "High"
            
            st.metric(
                label=f"{time_point}",
                value=f"{survival_pct:.1f}%",
                delta=f"Risk: {risk_pct:.1f}%"
            )
            
            st.markdown(f"<p class='{risk_class}'>Risk Level: {risk_label}</p>", 
                       unsafe_allow_html=True)

def load_sample_ecg():
    """
    Load a sample ECG from the HDF5 file for demonstration
    """
    hdf5_path = 'data/exams_part0.hdf5'
    
    if not os.path.exists(hdf5_path):
        st.warning("Sample HDF5 file not found. Please provide ECG data manually.")
        return None
    
    try:
        with h5py.File(hdf5_path, 'r') as hf:
            # Get first available ECG
            exam_ids = hf['exam_id'][:]
            if len(exam_ids) > 0:
                # Load first ECG and select required leads
                ecg_12lead = hf['tracings'][0]
                ecg_8lead = ecg_12lead[:, data_utils.LEAD_INDICES]
                
                return ecg_8lead, exam_ids[0]
    except Exception as e:
        st.error(f"Error loading sample ECG: {str(e)}")
    
    return None, None

def main():
    # Header
    st.title("❤️ ECG Mortality Risk Predictor")
    st.markdown("""
    This application uses a deep learning model trained on ECG data to predict 
    mortality risk over a 10-year period. The model analyzes 8-lead ECG signals
    and provides survival probability estimates at various time intervals.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        **Model Details:**
        - Architecture: 1D-CNN ResNet
        - Input: 8-lead ECG (4096 samples)
        - Output: 120 monthly survival probabilities
        - Training: Discrete-time survival analysis
        """)
        
        st.header("ECG Leads Used")
        st.write("The model uses the following 8 ECG leads:")
        for lead in data_utils.LEADS_REQUIRED:
            st.write(f"• {lead}")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Main content area
    st.header("📋 Patient ECG Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Use Sample ECG", "Upload ECG Data", "Manual Entry (Demo)"]
    )
    
    ecg_data = None
    exam_id = None
    
    if input_method == "Use Sample ECG":
        if st.button("Load Sample ECG"):
            with st.spinner("Loading sample ECG..."):
                ecg_data, exam_id = load_sample_ecg()
                
                if ecg_data is not None:
                    st.success(f"Sample ECG loaded successfully! (Exam ID: {exam_id})")
                    
                    # Display ECG shape info
                    st.write(f"ECG Shape: {ecg_data.shape}")
                    st.write(f"Duration: {ecg_data.shape[0]/400:.1f} seconds (assuming 400Hz)")
                else:
                    st.error("Failed to load sample ECG")
    
    elif input_method == "Upload ECG Data":
        st.info("Upload a numpy array file (.npy) containing ECG data with shape (4096, 8)")
        
        uploaded_file = st.file_uploader("Choose ECG file", type=['npy'])
        
        if uploaded_file is not None:
            try:
                ecg_data = np.load(uploaded_file)
                
                if ecg_data.shape == (4096, 8):
                    st.success("ECG data loaded successfully!")
                    st.write(f"ECG Shape: {ecg_data.shape}")
                else:
                    st.error(f"Invalid shape: {ecg_data.shape}. Expected (4096, 8)")
                    ecg_data = None
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif input_method == "Manual Entry (Demo)":
        st.info("Generating random ECG data for demonstration purposes")
        
        if st.button("Generate Demo ECG"):
            # Generate realistic-looking synthetic ECG data
            np.random.seed(42)
            t = np.linspace(0, 10.24, 4096)  # 10.24 seconds at 400Hz
            
            ecg_data = np.zeros((4096, 8))
            for lead in range(8):
                # Simulate ECG-like signal with different characteristics per lead
                base_freq = 1.2  # Heart rate ~72 bpm
                ecg_data[:, lead] = (
                    0.5 * np.sin(2 * np.pi * base_freq * t + np.random.rand() * 2 * np.pi) +
                    0.2 * np.sin(2 * np.pi * base_freq * 2 * t + np.random.rand() * 2 * np.pi) +
                    0.1 * np.sin(2 * np.pi * base_freq * 3 * t + np.random.rand() * 2 * np.pi) +
                    0.05 * np.random.randn(4096)
                )
            
            st.success("Demo ECG generated!")
            st.write(f"ECG Shape: {ecg_data.shape}")
    
    # Visualization of ECG signal (optional)
    if ecg_data is not None:
        with st.expander("View ECG Signal"):
            # Plot first 2 seconds of ECG for visualization
            fig_ecg = go.Figure()
            
            time_axis = np.arange(800) / 400  # First 2 seconds
            
            for i, lead in enumerate(data_utils.LEADS_REQUIRED[:4]):  # Show first 4 leads
                fig_ecg.add_trace(go.Scatter(
                    x=time_axis,
                    y=ecg_data[:800, i] + i * 2,  # Offset for clarity
                    mode='lines',
                    name=lead,
                    line=dict(width=1)
                ))
            
            fig_ecg.update_layout(
                title="ECG Signal Preview (First 2 seconds, 4 leads)",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude (offset for clarity)",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig_ecg, use_container_width=True)
    
    # Prediction section
    st.header("🔮 Mortality Risk Prediction")
    
    if ecg_data is not None:
        if st.button("Predict Risk", type="primary"):
            with st.spinner("Analyzing ECG and calculating risk..."):
                # Process ECG
                processed_ecg = process_ecg_input(ecg_data)
                
                if processed_ecg is not None:
                    # Make predictions
                    interval_probs, cumulative_survival = predict_survival(model, processed_ecg)
                    
                    # Calculate risk scores
                    risk_scores = calculate_risk_score(cumulative_survival)
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Risk assessment metrics
                    display_risk_assessment(risk_scores)
                    
                    # Survival curve
                    st.subheader("📈 Survival Probability Curve")
                    fig = plot_survival_curve(cumulative_survival)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results in expander
                    with st.expander("View Detailed Results"):
                        st.subheader("Monthly Survival Probabilities")
                        
                        # Create DataFrame for display
                        results_df = pd.DataFrame({
                            'Month': np.arange(N_INTERVALS),
                            'Year': np.arange(N_INTERVALS) / 12,
                            'Interval Survival': interval_probs,
                            'Cumulative Survival': cumulative_survival,
                            'Cumulative Risk': 1 - cumulative_survival
                        })
                        
                        # Format percentages
                        for col in ['Interval Survival', 'Cumulative Survival', 'Cumulative Risk']:
                            results_df[col] = (results_df[col] * 100).round(2)
                        
                        # Display key intervals
                        key_intervals = [5, 11, 23, 59, 119]  # 6mo, 1y, 2y, 5y, 10y
                        st.write("Key Time Points:")
                        st.dataframe(results_df.iloc[key_intervals])
                        
                        # Option to download full results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Full Results (CSV)",
                            data=csv,
                            file_name=f"ecg_risk_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    # Clinical interpretation guide
                    with st.expander("📚 Clinical Interpretation Guide"):
                        st.markdown("""
                        ### Understanding the Results
                        
                        **Survival Probability**: The likelihood that a patient will survive to a given time point.
                        
                        **Risk Categories**:
                        - **Low Risk** (>90% survival): Regular monitoring recommended
                        - **Medium Risk** (70-90% survival): Enhanced monitoring and preventive measures
                        - **High Risk** (<70% survival): Intensive monitoring and intervention recommended
                        
                        **Important Notes**:
                        - These predictions are based on ECG patterns and should be used in conjunction with other clinical assessments
                        - Individual patient factors (age, comorbidities, lifestyle) should be considered
                        - Regular follow-up and repeat assessments are recommended
                        
                        **Limitations**:
                        - Model trained on specific population data
                        - Predictions are probabilistic, not deterministic
                        - Should not be used as sole basis for clinical decisions
                        """)
    else:
        st.info("Please load or generate ECG data to perform risk prediction")
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer**: This tool is for research and educational purposes only. "
              "Always consult qualified healthcare professionals for medical decisions.")

if __name__ == "__main__":
    main()