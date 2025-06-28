import pytest
import sys

sys.path.append('.')

from train import train_and_evaluate

MINIMUM_ACCURACY = 0.85

@pytest.mark.skip(reason="This test requires a fully materialized Feast feature store and GCP auth.")
def test_model_performance():
    """
    Tests the model's performance against a minimum accuracy threshold.
    """
    print("Running model performance test...")
    
    accuracy = train_and_evaluate()
    
    print(f"Model accuracy from test: {accuracy:.3f}")
    
    assert accuracy >= MINIMUM_ACCURACY, \
        f"Model accuracy {accuracy:.3f} is below the threshold of {MINIMUM_ACCURACY}"