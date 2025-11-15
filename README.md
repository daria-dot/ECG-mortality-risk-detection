# ECG-mortality-risk-detection

This repository contains code for an ECG-based mortality risk prediction project, structured in phases.

- **Phase 1** (in `ecg_colab_project.ipynb`): Environment setup and data loading in Google Colab.
- **Phase 3** (in this repo via `label_engineering.py`): Discrete-time survival label engineering for 10-year mortality prediction.
- **Phase 4** (in `model_phase4.py`): 1D-CNN model for discrete-time survival prediction from ECG.

## Phase 3: Label Engineering

Phase 3 implements the discrete-time survival setup used in the Lancet-style ECG mortality work. It converts `exams.csv` data from the CODE-15% dataset into:

- `y_true` — survival labels, shape `(n_samples, 120)`
- `y_mask` — mask for valid intervals, shape `(n_samples, 120)`
- `exam_ids` — corresponding exam IDs

### Usage

```bash
pip install -r requirements.txt
```

```python
import pandas as pd
from label_engineering import create_survival_labels

# Load the CODE-15% exams.csv (Phase 1 puts this in your Drive)
exams_df = pd.read_csv("path/to/exams.csv")

# Optionally filter to rows with mortality data
exams_df = exams_df.dropna(subset=["death", "timey"])

# Create discrete-time survival labels (10 years, 120 intervals)
y_true, y_mask, exam_ids = create_survival_labels(
    exams_df,
    n_intervals=120,
    max_time_years=10.0,
    validate=True,
)
```

For more details, see `PHASE3_README.md` and `example_usage.py`. A comprehensive test suite is provided in `test_label_engineering.py`.

## Phase 4: 1D-CNN Survival Model

Phase 4 builds a 1D convolutional neural network that takes a single ECG tracing
and predicts the discrete-time survival probabilities for each interval,
matching the labels from Phase 3.

- **Input:** `X` with shape `(n_samples, 4096, 8)` (time, leads)
- **Output:** `N_INTERVALS` units (e.g. 120), each a sigmoid probability of surviving that interval
- **Loss:** Binary cross-entropy per interval, masked using `y_mask` from Phase 3 via `sample_weight`

### Basic Usage

```bash
pip install -r requirements.txt
```

```python
import numpy as np
from model_phase4 import compile_ecg_survival_model
from label_engineering import create_survival_labels

# 1. Load exams.csv and create labels (Phase 3)
exams_df = pd.read_csv("path/to/exams.csv")
exams_df = exams_df.dropna(subset=["death", "timey"])

y_true, y_mask, exam_ids = create_survival_labels(
    exams_df,
    n_intervals=120,
    max_time_years=10.0,
    validate=True,
)

# 2. Load or build your ECG input tensor X (Phase 2 output),
#    with shape (n_samples, 4096, 8)
X_train = ...  # your preprocessed ECGs

# 3. Build and compile the model (Phase 4)
model = compile_ecg_survival_model(
    n_intervals=120,
    input_length=4096,
    n_leads=8,
)

# 4. Train, passing the Phase 3 mask as sample_weight
history = model.fit(
    X_train,
    y_true,
    sample_weight=y_mask,
    epochs=50,
    batch_size=32,
)

# 5. Predict survival curves and risk scores (Phase 5 preview)
y_pred = model.predict(X_train)                 # (n_samples, 120)
survival_curves = np.cumprod(y_pred, axis=1)    # S(t_j)
risk_scores = 1 - survival_curves[:, -1]        # 10-year mortality risk
```
