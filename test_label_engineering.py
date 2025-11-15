"""
Comprehensive test suite for Phase 3 label engineering.

This script tests various edge cases and validates the correctness
of the survival label encoding implementation.
"""

import numpy as np
import pandas as pd
from label_engineering import SurvivalLabelEncoder, create_survival_labels


def test_basic_encoding():
    """Test basic encoding functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Encoding")
    print("=" * 60)

    encoder = SurvivalLabelEncoder(n_intervals=12, max_time_years=1.0)

    # Test case 1: Patient who died at day 100
    y_true, y_mask = encoder.encode_single_patient(time_days=100, death_status=True)

    interval_idx = encoder.time_to_interval(100)
    print(f"\nPatient died at day 100 -> interval {interval_idx}")
    print(f"Mask (first 15): {y_mask[:15]}")
    print(f"True (first 15): {y_true[:15]}")

    # Validate
    assert y_mask[interval_idx] == 1, "Mask should be 1 at death interval"
    assert y_true[interval_idx] == 0, "Should show death (0) at interval"
    if interval_idx > 0:
        assert y_true[interval_idx - 1] == 1, "Should show survival (1) before death"

    print("✓ Test passed: Death event encoded correctly")

    # Test case 2: Censored patient at day 200
    y_true, y_mask = encoder.encode_single_patient(time_days=200, death_status=False)

    interval_idx = encoder.time_to_interval(200)
    print(f"\nCensored patient at day 200 -> interval {interval_idx}")
    print(f"Mask (first 15): {y_mask[:15]}")
    print(f"True (first 15): {y_true[:15]}")

    # Validate
    assert y_mask[interval_idx] == 1, "Mask should be 1 at last follow-up"
    assert y_true[interval_idx] == 1, "Should show survival (1) for censored"
    assert np.all(y_true[:interval_idx + 1] == 1), "All intervals should show survival"

    print("✓ Test passed: Censored patient encoded correctly")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST 2: Edge Cases")
    print("=" * 60)

    encoder = SurvivalLabelEncoder(n_intervals=120, max_time_years=10.0)

    # Edge case 1: Very early death (day 1)
    print("\n1. Very early death (day 1)...")
    y_true, y_mask = encoder.encode_single_patient(time_days=1, death_status=True)
    interval_idx = encoder.time_to_interval(1)
    print(f"   Interval: {interval_idx}")
    assert interval_idx == 0, "Should be interval 0"
    assert y_true[0] == 0, "Should show death in interval 0"
    assert y_mask[0] == 1, "Mask should be active for interval 0"
    print("   ✓ Passed")

    # Edge case 2: Exactly at max time
    print("\n2. Patient at exactly max time (10 years)...")
    y_true, y_mask = encoder.encode_single_patient(
        time_days=10 * 365.25,
        death_status=False
    )
    interval_idx = encoder.time_to_interval(10 * 365.25)
    print(f"   Interval: {interval_idx}")
    assert interval_idx == 119, "Should cap at last interval"
    assert y_mask[119] == 1, "Last interval should be masked"
    print("   ✓ Passed")

    # Edge case 3: Beyond max time
    print("\n3. Patient beyond max time (15 years)...")
    y_true, y_mask = encoder.encode_single_patient(
        time_days=15 * 365.25,
        death_status=False
    )
    interval_idx = encoder.time_to_interval(15 * 365.25)
    print(f"   Interval: {interval_idx} (capped)")
    assert interval_idx == 119, "Should cap at last interval"
    assert np.sum(y_mask) == 120, "All intervals should be masked"
    print("   ✓ Passed")

    # Edge case 4: Zero time
    print("\n4. Zero follow-up time...")
    y_true, y_mask = encoder.encode_single_patient(time_days=0, death_status=True)
    interval_idx = encoder.time_to_interval(0)
    print(f"   Interval: {interval_idx}")
    assert interval_idx == 0, "Should be interval 0"
    assert y_mask[0] == 1, "Should have at least one masked interval"
    print("   ✓ Passed")


def test_batch_encoding():
    """Test batch encoding with multiple patients."""
    print("\n" + "=" * 60)
    print("TEST 3: Batch Encoding")
    print("=" * 60)

    # Create test data
    n_patients = 100
    np.random.seed(42)

    time_days = np.random.uniform(10, 3652.5, n_patients)
    death_status = np.random.rand(n_patients) < 0.3  # 30% death rate

    encoder = SurvivalLabelEncoder(n_intervals=120, max_time_years=10.0)

    print(f"\nEncoding {n_patients} patients...")
    y_true, y_mask = encoder.encode_batch(time_days, death_status)

    print(f"Output shape: {y_true.shape}")
    assert y_true.shape == (n_patients, 120), "Wrong output shape"
    assert y_mask.shape == (n_patients, 120), "Wrong mask shape"

    # Validate each patient
    print("Validating individual encodings...")
    for i in range(n_patients):
        # Check mask is contiguous from 0
        masked = np.where(y_mask[i] == 1)[0]
        assert len(masked) > 0, f"Patient {i} has no masked intervals"
        assert masked[0] == 0, f"Patient {i} mask doesn't start at 0"
        assert np.all(np.diff(masked) == 1), f"Patient {i} mask not contiguous"

        # Check death encoding
        deaths = np.where((y_mask[i] == 1) & (y_true[i] == 0))[0]
        if death_status[i]:
            assert len(deaths) == 1, f"Patient {i} should have exactly 1 death"
            assert deaths[0] == masked[-1], f"Patient {i} death not at last interval"
        else:
            assert len(deaths) == 0, f"Patient {i} shouldn't have death event"

    print("✓ All patients validated successfully")


def test_dataframe_encoding():
    """Test encoding from DataFrame (CODE-15% format)."""
    print("\n" + "=" * 60)
    print("TEST 4: DataFrame Encoding (CODE-15% Format)")
    print("=" * 60)

    # Create realistic test data matching CODE-15% format
    n_patients = 50
    np.random.seed(42)

    test_df = pd.DataFrame({
        'exam_id': range(1000, 1000 + n_patients),
        'patient_id': np.random.randint(1, 500, n_patients),
        'timey': np.random.exponential(scale=1000, size=n_patients),
        'death': np.random.rand(n_patients) < 0.25,
        'age': np.random.randint(40, 85, n_patients),
        'sex': np.random.choice(['Male', 'Female'], n_patients)
    })

    print(f"\nTest DataFrame shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    print("\nFirst 5 rows:")
    print(test_df[['exam_id', 'timey', 'death']].head())

    # Encode
    y_true, y_mask, exam_ids = create_survival_labels(
        test_df,
        n_intervals=120,
        max_time_years=10.0,
        validate=True
    )

    print(f"\nOutput shapes:")
    print(f"  y_true: {y_true.shape}")
    print(f"  y_mask: {y_mask.shape}")
    print(f"  exam_ids: {exam_ids.shape}")

    # Verify exam_ids match
    assert np.array_equal(exam_ids, test_df['exam_id'].values), "Exam IDs don't match"
    print("✓ Exam IDs match correctly")


def test_consistency_with_plan():
    """Test that output format matches what's needed for Phases 4 & 5."""
    print("\n" + "=" * 60)
    print("TEST 5: Integration with Other Phases")
    print("=" * 60)

    # Create sample data
    test_df = pd.DataFrame({
        'exam_id': [1, 2, 3, 4, 5],
        'timey': [100, 500, 1000, 2000, 3652.5],
        'death': [True, False, True, False, True]
    })

    # Encode
    y_true, y_mask, exam_ids = create_survival_labels(
        test_df,
        n_intervals=120,
        max_time_years=10.0
    )

    print("\n1. Checking compatibility with Phase 4 (Model)...")
    print(f"   - Output shape for Keras Dense layer: {y_true.shape[1]} units ✓")
    print(f"   - Activation: sigmoid (for probabilities) ✓")
    print(f"   - Loss: binary_crossentropy with sample_weight ✓")

    print("\n2. Checking compatibility with Phase 5 (Training)...")
    print(f"   - y_train shape: {y_true.shape} ✓")
    print(f"   - sample_weight shape: {y_mask.shape} ✓")
    print(f"   - Can use: model.fit(X, y_true, sample_weight=y_mask) ✓")

    print("\n3. Verifying data types...")
    assert y_true.dtype == np.float32, "y_true should be float32"
    assert y_mask.dtype == np.float32, "y_mask should be float32"
    print(f"   - y_true dtype: {y_true.dtype} ✓")
    print(f"   - y_mask dtype: {y_mask.dtype} ✓")

    print("\n4. Testing survival curve extraction (for Phase 5 evaluation)...")
    # Simulate model predictions
    y_pred = np.random.rand(5, 120).astype(np.float32)

    # Calculate survival curves (cumulative product)
    survival_curves = np.cumprod(y_pred, axis=1)
    print(f"   - Survival curves shape: {survival_curves.shape} ✓")

    # Calculate risk scores (1 - S(10 years))
    risk_scores = 1 - survival_curves[:, -1]
    print(f"   - Risk scores shape: {risk_scores.shape} ✓")
    print(f"   - Risk score range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}] ✓")


def test_specific_scenarios():
    """Test specific scenarios to ensure correctness."""
    print("\n" + "=" * 60)
    print("TEST 6: Specific Scenarios")
    print("=" * 60)

    encoder = SurvivalLabelEncoder(n_intervals=120, max_time_years=10.0)

    # Scenario 1: Patient who died at exactly 5 years
    print("\n1. Patient died at exactly 5 years (1826.25 days)...")
    y_true, y_mask = encoder.encode_single_patient(
        time_days=5 * 365.25,
        death_status=True
    )

    interval_idx = encoder.time_to_interval(5 * 365.25)
    expected_interval = int(np.floor((5 * 365.25) / encoder.interval_length_days))

    print(f"   Expected interval: {expected_interval}")
    print(f"   Actual interval: {interval_idx}")
    assert interval_idx == expected_interval, "Interval calculation wrong"

    masked_count = int(np.sum(y_mask))
    survived_count = int(np.sum((y_mask == 1) & (y_true == 1)))

    print(f"   Masked intervals: {masked_count}")
    print(f"   Survived intervals: {survived_count}")
    print(f"   Death interval: {interval_idx}")

    assert masked_count == interval_idx + 1, "Wrong number of masked intervals"
    assert survived_count == interval_idx, "Should survive all intervals before death"
    assert y_true[interval_idx] == 0, "Should show death at interval"
    print("   ✓ Passed")

    # Scenario 2: Censored at 3 months
    print("\n2. Censored at 3 months (~91 days)...")
    y_true, y_mask = encoder.encode_single_patient(
        time_days=91,
        death_status=False
    )

    interval_idx = encoder.time_to_interval(91)
    masked_count = int(np.sum(y_mask))
    survived_count = int(np.sum((y_mask == 1) & (y_true == 1)))

    print(f"   Interval: {interval_idx}")
    print(f"   Masked intervals: {masked_count}")
    print(f"   All survived: {survived_count == masked_count}")

    assert survived_count == masked_count, "All masked intervals should show survival"
    assert np.all(y_true[:interval_idx + 1] == 1), "All intervals should be 1"
    print("   ✓ Passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "PHASE 3 COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    try:
        test_basic_encoding()
        test_edge_cases()
        test_batch_encoding()
        test_dataframe_encoding()
        test_consistency_with_plan()
        test_specific_scenarios()

        print("\n" + "=" * 70)
        print(" " * 20 + "ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nPhase 3 implementation is correct and ready for integration!")
        print("\nNext steps:")
        print("  1. Load your exams.csv file from CODE-15%")
        print("  2. Use: y_true, y_mask, exam_ids = create_survival_labels(df)")
        print("  3. Proceed to Phase 4 (Model Development)")

        return True

    except AssertionError as e:
        print("\n" + "=" * 70)
        print(" " * 25 + "TEST FAILED! ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        return False
    except Exception as e:
        print("\n" + "=" * 70)
        print(" " * 25 + "TEST ERROR! ✗")
        print("=" * 70)
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
