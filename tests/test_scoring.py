"""
Tests for the scoring module.
"""
import pytest
from functions import scoring


@pytest.fixture
def test_pdb_path():
    """
    Fixture to provide the test PDB path.
    """
    return "tests/test_complex.pdb"


def test_score_interface(test_pdb_path):
    """
    Test the score_interface function.
    """
    scores, _, _ = scoring.score_interface(test_pdb_path, binder_chain="B")
    assert isinstance(scores, dict)
    assert "interface_nres" in scores
    