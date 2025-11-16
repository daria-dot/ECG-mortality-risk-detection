# ECG Mortality Risk Detection - Streamlit Application

## Overview
This Streamlit application provides a user-friendly interface for predicting mortality risk from ECG data using a trained deep learning model. The model analyzes 8-lead ECG signals and provides survival probability estimates over a 10-year period.

## Features
- **Interactive Web Interface**: User-friendly Streamlit application
- **Multiple Input Methods**: 
  - Load sample ECG from HDF5 file
  - Upload ECG data as numpy array
  - Generate demo ECG for testing
- **Risk Visualization**: 
  - Risk assessment metrics at key time points (6 months, 1 year, 2 years, 5 years, 10 years)
  - Interactive survival probability curve
  - ECG signal visualization
- **Detailed Results**: 
  - Monthly survival probabilities
  - Downloadable CSV reports
  - Clinical interpretation guide

## Prerequisites

### 1. Trained Model
Ensure you have the trained model saved at `data/ecg_survival_model.h5`. If not, run the training pipeline first:
```bash
python main.py
```

### 2. Dependencies
Install required packages:
```bash
pip install -r requirements_app.txt
```

## Running the Application

### Basic Usage
1. Navigate to the project directory:
```bash
cd ECG-mortality-risk-detection
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your web browser (typically at http://localhost:8501)

### Using the Application

1. **Load ECG Data**: Choose one of three methods:
   - **Use Sample ECG**: Loads the first ECG from the HDF5 file
   - **Upload ECG Data**: Upload a `.npy` file containing ECG data (shape: 4096 x 8)
   - **Manual Entry (Demo)**: Generates synthetic ECG data for demonstration

2. **Analyze ECG**: Click "Predict Risk" to run the model

3. **View Results**:
   - **Risk Assessment**: Shows survival probability and risk level at key time points
   - **Survival Curve**: Interactive plot showing probability over 10 years
   - **Detailed Results**: View monthly probabilities and download CSV report

## Input Data Format

### ECG Data Requirements
- **Shape**: (4096, 8) - 4096 time samples across 8 ECG leads
- **Leads Used**: DI, DII, V1, V2, V3, V4, V5, V6
- **Sampling Rate**: Assumed 400 Hz (10.24 seconds of data)
- **Data Type**: Float32 numpy array

### Creating Compatible ECG Files
To save ECG data for upload:
```python
import numpy as np

# Your ECG data should be shape (4096, 8)
ecg_data = np.random.randn(4096, 8).astype(np.float32)

# Save for upload to app
np.save('patient_ecg.npy', ecg_data)
```

## Model Details

### Architecture
- **Type**: 1D-CNN ResNet
- **Input**: 8-lead ECG (4096 samples per lead)
- **Output**: 120 monthly survival probabilities
- **Training Method**: Discrete-time survival analysis

### Prediction Outputs
- **Interval Survival Probabilities**: Probability of surviving each month
- **Cumulative Survival**: Overall survival probability at each time point
- **Risk Scores**: 1 - survival probability

## Risk Categories

The app categorizes risk based on survival probability:
- **Low Risk** (>90% survival): Regular monitoring recommended
- **Medium Risk** (70-90% survival): Enhanced monitoring and preventive measures
- **High Risk** (<70% survival): Intensive monitoring and intervention recommended

## Troubleshooting

### Common Issues

1. **Model not found error**: 
   - Ensure the model is trained and saved at `data/ecg_survival_model.h5`
   - Run `python main.py` to train the model if needed

2. **Import errors**:
   - Install all dependencies: `pip install -r requirements_app.txt`
   - Ensure you're in the correct directory with all project files

3. **ECG data format errors**:
   - Verify data shape is exactly (4096, 8)
   - Ensure data is float32 type
   - Check that all 8 required leads are present

### Advanced Configuration

To modify the app behavior, edit these constants in `app.py`:
```python
MODEL_PATH = 'data/ecg_survival_model.h5'  # Path to saved model
N_INTERVALS = 120  # Number of prediction intervals
MAX_TIME_YEARS = 10  # Maximum prediction time in years
```

## API Usage (Alternative)

For programmatic access without the UI:
```python
import numpy as np
import tensorflow as tf
from app import load_model, process_ecg_input, predict_survival, calculate_risk_score

# Load model
model = tf.keras.models.load_model('data/ecg_survival_model.h5', 
                                   custom_objects={'survival_loss_with_mask': ...})

# Prepare ECG data
ecg_data = np.load('patient_ecg.npy')  # Shape: (4096, 8)
processed = process_ecg_input(ecg_data)

# Make predictions
interval_probs, cumulative_survival = predict_survival(model, processed)
risk_scores = calculate_risk_score(cumulative_survival)

print(f"10-year survival probability: {cumulative_survival[-1]:.2%}")
print(f"10-year mortality risk: {1 - cumulative_survival[-1]:.2%}")
```

## Disclaimer
⚠️ **Important**: This tool is for research and educational purposes only. The predictions should not be used as the sole basis for clinical decisions. Always consult qualified healthcare professionals for medical advice and treatment decisions.

## Support
For issues or questions about the application, please refer to the main project documentation or open an issue in the repository.