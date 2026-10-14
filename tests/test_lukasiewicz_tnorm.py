import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablation_ladder_rolling_origin import compute_customer_concept_memberships


def _concepts(intent):
    return pd.DataFrame([{"intent": intent, "intent_size": len(intent.split(" & "))}])


def test_single_attribute_is_identity():
    got = compute_customer_concept_memberships(
        _concepts("R1"), pd.DataFrame({"R1": [0.0, 0.3, 1.0]}), tnorm="lukasiewicz"
    )
    np.testing.assert_allclose(got[:, 0], [0.0, 0.3, 1.0])


def test_zero_membership_clips_to_zero():
    got = compute_customer_concept_memberships(
        _concepts("R1 & F2 & M3"),
        pd.DataFrame({"R1": [0.9, 0.0], "F2": [0.8, 0.9], "M3": [0.7, 0.9]}),
        tnorm="lukasiewicz",
    )
    np.testing.assert_allclose(got[:, 0], [0.4, 0.0])


def test_boundary_and_saturation_values():
    got = compute_customer_concept_memberships(
        _concepts("R1 & F2"),
        pd.DataFrame({"R1": [0.0, 0.5, 1.0], "F2": [1.0, 0.5, 1.0]}),
        tnorm="lukasiewicz",
    )
    np.testing.assert_allclose(got[:, 0], [0.0, 0.0, 1.0])


def test_unknown_tnorm_is_rejected():
    try:
        compute_customer_concept_memberships(_concepts("R1"), pd.DataFrame({"R1": [1.0]}), tnorm="bad")
    except ValueError:
        return
    raise AssertionError("unsupported t-norm was accepted")
