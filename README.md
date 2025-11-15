# ECG-mortality-risk-detection

This repository contains code for an ECG-based mortality risk prediction project, structured in phases.

- **Phase 1** (in `ecg_colab_project.ipynb`): Environment setup and data loading in Google Colab.
- **Phase 3** (in this repo via `label_engineering.py`): Discrete-time survival label engineering for 10-year mortality prediction.

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
