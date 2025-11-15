"""
Phase 3: Label Engineering for Discrete-Time Survival Model

This module converts survival data (death status and follow-up time) into
the discrete-time format required for training a deep learning survival model.

The implementation follows the methodology from the Lancet paper, creating:
1. y_true: Binary survival labels for each time interval
2. y_mask: Mask indicating which intervals are valid for each patient

Author: Biohack Project
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings


class SurvivalLabelEncoder:
    """
    Encodes survival data into discrete-time format for deep learning models.

    This encoder converts continuous survival times and event indicators into
    discrete interval-based labels suitable for training neural networks.

    Attributes:
        n_intervals (int): Number of discrete time intervals
        max_time_days (float): Maximum follow-up time in days
        interval_length_days (float): Length of each interval in days
    """

    def __init__(
        self,
        n_intervals: int = 120,
        max_time_years: float = 10.0,
        verbose: bool = True,
    ):
        """
        Initialize the survival label encoder.

        Args:
            n_intervals: Number of discrete time intervals (default: 120 for monthly intervals)
            max_time_years: Maximum follow-up period in years (default: 10 years)
        """
        self.n_intervals = n_intervals
        self.max_time_days = max_time_years * 365.25
        self.interval_length_days = self.max_time_days / n_intervals
        self.verbose = verbose

        if self.verbose:
            print(f"SurvivalLabelEncoder initialized:")
            print(f"  - Number of intervals: {self.n_intervals}")
            print(f"  - Max time: {max_time_years} years ({self.max_time_days:.1f} days)")
            print(f"  - Interval length: {self.interval_length_days:.2f} days (~{self.interval_length_days/30.4:.2f} months)")

    def time_to_interval(self, time_days: float) -> int:
        """
        Convert continuous time (in days) to discrete interval index.

        Args:
            time_days: Follow-up time in days

        Returns:
            Interval index (0 to n_intervals-1)
        """
        # Calculate interval index
        interval_idx = int(np.floor(time_days / self.interval_length_days))

        # Cap at maximum interval
        interval_idx = min(interval_idx, self.n_intervals - 1)

        # Ensure non-negative
        interval_idx = max(0, interval_idx)

        return interval_idx

    def encode_single_patient(
        self,
        time_days: float,
        death_status: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a single patient's survival data.

        Args:
            time_days: Follow-up time in days
            death_status: True if patient died, False if censored

        Returns:
            Tuple of (y_true, y_mask) arrays, each of shape (n_intervals,)

        Logic:
            - Interval j is when the patient was last seen (or died)
            - y_mask[0:j+1] = 1 (we care about intervals up to j)
            - If censored (death=False): y_true[0:j+1] = 1 (survived all intervals)
            - If died (death=True):
                - y_true[0:j] = 1 (survived intervals before j)
                - y_true[j] = 0 (died in interval j)
        """
        # Initialize arrays
        y_true = np.zeros(self.n_intervals, dtype=np.float32)
        y_mask = np.zeros(self.n_intervals, dtype=np.float32)

        # Get the interval index for this patient
        j = self.time_to_interval(time_days)

        # Set the mask (intervals we care about)
        y_mask[0:j+1] = 1.0

        # Set the true labels based on death status
        if death_status:
            # Patient died in interval j
            if j > 0:
                y_true[0:j] = 1.0  # Survived all intervals before j
            y_true[j] = 0.0  # Died in interval j
        else:
            # Patient was censored (survived or lost to follow-up)
            y_true[0:j+1] = 1.0  # Survived all observed intervals

        return y_true, y_mask

    def encode_batch(
        self,
        time_days_array: np.ndarray,
        death_status_array: np.ndarray,
        exam_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode multiple patients' survival data in batch.

        Args:
            time_days_array: Array of follow-up times in days, shape (n_samples,)
            death_status_array: Array of death statuses (bool or 0/1), shape (n_samples,)
            exam_ids: Optional array of exam IDs for tracking

        Returns:
            Tuple of (y_true, y_mask) arrays, each of shape (n_samples, n_intervals)
        """
        n_samples = len(time_days_array)

        # Validate inputs
        if len(death_status_array) != n_samples:
            raise ValueError(
                f"Length mismatch: time_days_array has {n_samples} samples, "
                f"but death_status_array has {len(death_status_array)}"
            )

        # Initialize output arrays
        y_true = np.zeros((n_samples, self.n_intervals), dtype=np.float32)
        y_mask = np.zeros((n_samples, self.n_intervals), dtype=np.float32)

        # Encode each patient
        for i in range(n_samples):
            y_true[i], y_mask[i] = self.encode_single_patient(
                time_days=time_days_array[i],
                death_status=bool(death_status_array[i])
            )

        return y_true, y_mask

    def encode_from_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = 'timey',
        death_col: str = 'death',
        exam_id_col: str = 'exam_id'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode survival data directly from a pandas DataFrame.

        This is the primary method for use with the CODE-15% dataset.

        Args:
            df: DataFrame containing survival data (e.g., exams.csv)
            time_col: Column name for follow-up time in days (default: 'timey')
            death_col: Column name for death status (default: 'death')
            exam_id_col: Column name for exam IDs (default: 'exam_id')

        Returns:
            Tuple of (y_true, y_mask, exam_ids)
            - y_true: shape (n_samples, n_intervals)
            - y_mask: shape (n_samples, n_intervals)
            - exam_ids: shape (n_samples,)
        """
        # Validate columns exist
        required_cols = [time_col, death_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Extract data
        time_days = df[time_col].values
        # Ensure death status is boolean for correct statistics and encoding
        death_status = df[death_col].astype(bool).values
        exam_ids = df[exam_id_col].values if exam_id_col in df.columns else None

        # Compute statistics
        events = int(death_status.sum())
        censored = len(df) - events
        # Print statistics
        if self.verbose:
            print(f"\nEncoding survival labels for {len(df)} patients:")
            print(f"  - Events (deaths): {events} ({100 * events / len(df):.2f}%)")
            print(f"  - Censored: {censored} ({100 * censored / len(df):.2f}%)")
            print(f"  - Mean follow-up: {time_days.mean():.1f} days ({time_days.mean()/365.25:.2f} years)")
            print(f"  - Median follow-up: {np.median(time_days):.1f} days ({np.median(time_days)/365.25:.2f} years)")
            print(f"  - Min follow-up: {time_days.min():.1f} days")
            print(f"  - Max follow-up: {time_days.max():.1f} days")

        # Check for data quality issues
        if np.any(time_days < 0):
            warnings.warn(f"Found {(time_days < 0).sum()} negative follow-up times")

        if np.any(time_days > self.max_time_days):
            n_over = (time_days > self.max_time_days).sum()
            warnings.warn(
                f"Found {n_over} patients with follow-up > {self.max_time_days:.0f} days. "
                f"These will be capped at interval {self.n_intervals-1}."
            )

        # Encode
        y_true, y_mask = self.encode_batch(time_days, death_status, exam_ids)

        return y_true, y_mask, exam_ids

    def validate_encoding(
        self,
        y_true: np.ndarray,
        y_mask: np.ndarray,
        verbose: bool = True
    ) -> bool:
        """
        Validate that encoded labels are correct.

        Args:
            y_true: True labels array, shape (n_samples, n_intervals)
            y_mask: Mask array, shape (n_samples, n_intervals)
            verbose: Print validation details

        Returns:
            True if validation passes, False otherwise
        """
        n_samples = y_true.shape[0]

        if verbose:
            print("\n=== Validation Results ===")

        # Check shapes
        if y_true.shape != y_mask.shape:
            print(f"❌ Shape mismatch: y_true {y_true.shape} != y_mask {y_mask.shape}")
            return False

        if y_true.shape[1] != self.n_intervals:
            print(f"❌ Wrong number of intervals: {y_true.shape[1]} != {self.n_intervals}")
            return False

        # Check data types and values
        if not np.all((y_true >= 0) & (y_true <= 1)):
            print("❌ y_true contains values outside [0, 1]")
            return False

        if not np.all((y_mask == 0) | (y_mask == 1)):
            print("❌ y_mask contains values other than 0 or 1")
            return False

        # Check logical consistency
        issues = 0
        for i in range(n_samples):
            # Find where mask is active
            mask_active = np.where(y_mask[i] == 1)[0]

            if len(mask_active) == 0:
                print(f"❌ Sample {i}: No active mask intervals")
                issues += 1
                continue

            # Mask should be contiguous from 0
            if mask_active[0] != 0:
                print(f"❌ Sample {i}: Mask doesn't start at interval 0")
                issues += 1

            if not np.all(np.diff(mask_active) == 1):
                print(f"❌ Sample {i}: Mask is not contiguous")
                issues += 1

            # Check for death events (y_true = 0 within masked region)
            deaths_in_mask = np.where((y_mask[i] == 1) & (y_true[i] == 0))[0]

            if len(deaths_in_mask) > 0:
                # Should only have one death, and it should be the last masked interval
                last_mask_idx = mask_active[-1]

                if len(deaths_in_mask) > 1:
                    print(f"❌ Sample {i}: Multiple death events: {deaths_in_mask}")
                    issues += 1

                if deaths_in_mask[0] != last_mask_idx:
                    print(f"❌ Sample {i}: Death not at last masked interval")
                    issues += 1

        if issues > 0:
            print(f"❌ Found {issues} validation issues")
            return False

        if verbose:
            # Calculate statistics
            n_events = np.sum(np.any((y_mask == 1) & (y_true == 0), axis=1))
            n_censored = n_samples - n_events
            avg_intervals = np.sum(y_mask, axis=1).mean()

            print(f"✓ Shape: {y_true.shape}")
            print(f"✓ Events: {n_events} ({100*n_events/n_samples:.2f}%)")
            print(f"✓ Censored: {n_censored} ({100*n_censored/n_samples:.2f}%)")
            print(f"✓ Average masked intervals: {avg_intervals:.1f}")
            print(f"✓ All validation checks passed!")

        return True


def create_survival_labels(
    exams_df: pd.DataFrame,
    n_intervals: int = 120,
    max_time_years: float = 10.0,
    validate: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to create survival labels from exams DataFrame.

    This is the main entry point for Phase 3 label engineering.

    Args:
        exams_df: DataFrame from exams.csv with columns 'timey', 'death', 'exam_id'
        n_intervals: Number of discrete time intervals (default: 120)
        max_time_years: Maximum follow-up period in years (default: 10)
        validate: Whether to validate the encoded labels (default: True)

    Returns:
        Tuple of (y_true, y_mask, exam_ids)

    Example:
        >>> import pandas as pd
        >>> exams_df = pd.read_csv('exams.csv')
        >>> y_true, y_mask, exam_ids = create_survival_labels(exams_df)
        >>> print(f"Labels created: {y_true.shape}")
    """
    # Create encoder
    encoder = SurvivalLabelEncoder(
        n_intervals=n_intervals,
        max_time_years=max_time_years,
        verbose=verbose,
    )

    # Encode labels
    y_true, y_mask, exam_ids = encoder.encode_from_dataframe(exams_df)

    # Validate if requested
    if validate:
        is_valid = encoder.validate_encoding(y_true, y_mask, verbose=True)
        if not is_valid:
            raise ValueError("Label encoding validation failed!")

    return y_true, y_mask, exam_ids


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("=" * 60)
    print("Phase 3: Label Engineering - Example Usage")
    print("=" * 60)

    # Create synthetic example data
    print("\n1. Testing with synthetic data...")

    synthetic_data = pd.DataFrame({
        'exam_id': [1, 2, 3, 4, 5],
        'timey': [100, 500, 1000, 3000, 3652.5],  # days
        'death': [True, False, True, False, True]
    })

    print("\nSynthetic patient data:")
    print(synthetic_data)

    # Create labels
    y_true, y_mask, exam_ids = create_survival_labels(
        synthetic_data,
        n_intervals=120,
        max_time_years=10.0
    )

    print("\n2. Examining individual patient encodings...")
    for i, exam_id in enumerate(exam_ids):
        time_days = synthetic_data.loc[i, 'timey']
        died = synthetic_data.loc[i, 'death']

        masked_intervals = np.where(y_mask[i] == 1)[0]
        death_intervals = np.where((y_mask[i] == 1) & (y_true[i] == 0))[0]

        print(f"\nPatient {exam_id}:")
        print(f"  Time: {time_days:.1f} days ({time_days/365.25:.2f} years)")
        print(f"  Died: {died}")
        print(f"  Masked intervals: {len(masked_intervals)} (0 to {masked_intervals[-1]})")
        if len(death_intervals) > 0:
            print(f"  Death occurred in interval: {death_intervals[0]}")
        else:
            print(f"  Censored (survived all {len(masked_intervals)} intervals)")

    print("\n" + "=" * 60)
    print("Phase 3 implementation complete and validated!")
    print("=" * 60)
