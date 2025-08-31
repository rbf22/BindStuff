"""Tests for the colabdesign_utils module."""
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from functions import colabdesign_utils


def test_get_best_plddt():
    """Test the get_best_plddt function."""
    af_model = MagicMock()
    af_model._tmp = {"best": {"aux": {"plddt": np.array([80, 90, 95])}}}
    plddt = colabdesign_utils.get_best_plddt(af_model, 3)
    assert isinstance(plddt, float)
    assert plddt == 88.33

    # Test with empty plddt
    af_model._tmp = {"best": {"aux": {"plddt": np.array([])}}}
    plddt = colabdesign_utils.get_best_plddt(af_model, 0)
    assert np.isnan(plddt)


def test_add_rg_loss():
    """Test the add_rg_loss function."""
    af_model = MagicMock()
    af_model._callbacks = {"model": {"loss": []}}
    af_model.opt = {"weights": {}}
    colabdesign_utils.add_rg_loss(af_model, weight=0.5)
    assert "rg" in af_model.opt["weights"]
    assert len(af_model._callbacks["model"]["loss"]) == 1
    assert af_model.opt["weights"]["rg"] == 0.5

def test_add_i_ptm_loss():
    """Test the add_i_ptm_loss function."""
    af_model = MagicMock()
    af_model._callbacks = {"model": {"loss": []}}
    af_model.opt = {"weights": {}}
    colabdesign_utils.add_i_ptm_loss(af_model, weight=0.5)
    assert "i_ptm" in af_model.opt["weights"]
    assert len(af_model._callbacks["model"]["loss"]) == 1
    assert af_model.opt["weights"]["i_ptm"] == 0.5

def test_add_helix_loss():
    """Test the add_helix_loss function."""
    af_model = MagicMock()
    af_model._callbacks = {"model": {"loss": []}}
    af_model.opt = {"weights": {}}
    colabdesign_utils.add_helix_loss(af_model, weight=0.5)
    assert "helix" in af_model.opt["weights"]
    assert len(af_model._callbacks["model"]["loss"]) == 1
    assert af_model.opt["weights"]["helix"] == 0.5

def test_add_termini_distance_loss():
    """Test the add_termini_distance_loss function."""
    af_model = MagicMock()
    af_model._callbacks = {"model": {"loss": []}}
    af_model.opt = {"weights": {}}
    colabdesign_utils.add_termini_distance_loss(af_model, weight=0.5)
    assert "NC" in af_model.opt["weights"]
    assert len(af_model._callbacks["model"]["loss"]) == 1
    assert af_model.opt["weights"]["NC"] == 0.5

@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_plot_trajectory(mock_close, mock_savefig):
    """Test the plot_trajectory function."""
    af_model = MagicMock()
    af_model.get_loss.return_value = [0.1, 0.2, 0.3]
    af_model.aux = {"log": {"loss": [0.1, 0.2, 0.3]}}
    with tempfile.TemporaryDirectory() as tmpdir:
        design_paths = {"Trajectory/Plots": tmpdir}
        colabdesign_utils.plot_trajectory(af_model, "test_design", design_paths)
        mock_savefig.assert_called()
        mock_close.assert_called()
