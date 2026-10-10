"""Canonical naming for the existing LAR-PW membership-ablation arm.

This module contains identity metadata only.  The membership mathematics remains
implemented in :mod:`tail_memberships`, and the historical result files retain
their original internal arm identifier ``M1``.
"""

CANONICAL_METHOD = "M1-LAR-PW"
HISTORICAL_INTERNAL_METHOD = "M1"
HISTORICAL_MEMBERSHIP_TYPE = "lar_pw"

METHOD_MAPPING = {
    HISTORICAL_INTERNAL_METHOD: CANONICAL_METHOD,
}


def canonical_method_name(identifier: str) -> str:
    """Return the canonical label for a historical or canonical identifier."""
    if identifier == CANONICAL_METHOD:
        return CANONICAL_METHOD
    if identifier == HISTORICAL_INTERNAL_METHOD:
        return METHOD_MAPPING[identifier]
    raise ValueError(f"Unknown LAR-PW method identifier: {identifier}")
