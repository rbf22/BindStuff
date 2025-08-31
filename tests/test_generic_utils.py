"""Tests for the generic_utils module."""
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import pandas as pd

from functions import generic_utils


def test_generate_dataframe_labels():
    """Test the generate_dataframe_labels function."""
    trajectory_labels, design_labels, final_labels = generic_utils.generate_dataframe_labels()
    assert isinstance(trajectory_labels, list)
    assert isinstance(design_labels, list)
    assert isinstance(final_labels, list)
    assert "Design" in trajectory_labels
    assert "Average_pLDDT" in design_labels
    assert "Rank" in final_labels

def test_generate_directories():
    """Test the generate_directories function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        design_paths = generic_utils.generate_directories(tmpdir)
        assert isinstance(design_paths, dict)
        for name, path in design_paths.items():
            assert os.path.isdir(path)
            assert name in path

def test_generate_filter_pass_csv():
    """Test the generate_filter_pass_csv function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        failure_csv = os.path.join(tmpdir, "failures.csv")
        filter_json = os.path.join(tmpdir, "filters.json")
        with open(filter_json, "w", encoding="utf-8") as f:
            json.dump({"1_pLDDT": {"threshold": 0.7}}, f)
        generic_utils.generate_filter_pass_csv(failure_csv, filter_json)
        assert os.path.exists(failure_csv)
        df = pd.read_csv(failure_csv)
        assert "pLDDT" in df.columns

def test_update_failures():
    """Test the update_failures function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        failure_csv = os.path.join(tmpdir, "failures.csv")
        df = pd.DataFrame({"col1": [0], "col2": [0]})
        df.to_csv(failure_csv, index=False)
        generic_utils.update_failures(failure_csv, "col1")
        df = pd.read_csv(failure_csv)
        assert df["col1"][0] == 1
        generic_utils.update_failures(failure_csv, {"col2": 2})
        df = pd.read_csv(failure_csv)
        assert df["col2"][0] == 2

def test_check_n_trajectories():
    """Test the check_n_trajectories function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        design_paths = {"Trajectory/Relaxed": tmpdir}
        advanced_settings = {"max_trajectories": 2}
        assert not generic_utils.check_n_trajectories(design_paths, advanced_settings)
        with open(os.path.join(tmpdir, "file1.pdb"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(tmpdir, "file2.pdb"), "w", encoding="utf-8") as f:
            f.write("")
        assert generic_utils.check_n_trajectories(design_paths, advanced_settings)

def test_load_helicity():
    """Test the load_helicity function."""
    adv_set = {"random_helicity": False, "weights_helicity": 0.5}
    assert generic_utils.load_helicity(adv_set) == 0.5
    adv_set = {"random_helicity": True, "weights_helicity": 0}
    assert isinstance(generic_utils.load_helicity(adv_set), float)

@patch("jax.devices")
def test_check_jax_gpu(mock_devices):
    """Test the check_jax_gpu function."""
    mock_devices.return_value = [MagicMock(platform="gpu")]
    with patch("builtins.print") as mock_print:
        generic_utils.check_jax_gpu()
        mock_print.assert_any_call("Available GPUs:")

def test_perform_input_check():
    """Test the perform_input_check function."""
    args = MagicMock()
    args.settings = "settings.json"
    args.filters = None
    args.advanced = None
    _, f, a = generic_utils.perform_input_check(args)
    assert f is not None
    assert a is not None

def test_perform_advanced_settings_check():
    """Test the perform_advanced_settings_check function."""
    adv_set = {
        "af_params_dir": None, "dssp_path": None,
        "dalphaball_path": None, "omit_AAs": " C "
    }
    adv_set = generic_utils.perform_advanced_settings_check(adv_set, "test_folder")
    assert adv_set["af_params_dir"] == "test_folder"
    assert adv_set["omit_AAs"] == "C"

def test_load_af2_models():
    """Test the load_af2_models function."""
    d, p, m = generic_utils.load_af2_models(True)
    assert len(d) == 5
    assert len(p) == 2
    assert not m
    d, p, m = generic_utils.load_af2_models(False)
    assert len(d) == 2
    assert len(p) == 5
    assert m

def test_create_dataframe():
    """Test the create_dataframe function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test.csv")
        columns = ["col1", "col2"]
        generic_utils.create_dataframe(csv_file, columns)
        assert os.path.exists(csv_file)
        df = pd.read_csv(csv_file)
        assert list(df.columns) == columns

def test_insert_data():
    """Test the insert_data function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test.csv")
        columns = ["col1", "col2"]
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
        data = [1, "a"]
        generic_utils.insert_data(csv_file, data)
        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert df["col1"][0] == 1

def test_save_fasta():
    """Test the save_fasta function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        design_paths = {"MPNN/Sequences": tmpdir}
        generic_utils.save_fasta("design1", "ACGT", design_paths)
        fasta_file = os.path.join(tmpdir, "design1.fasta")
        assert os.path.exists(fasta_file)
        with open(fasta_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert lines[0] == ">design1\n"
            assert lines[1] == "ACGT\n"

def test_clean_pdb():
    """Test the clean_pdb function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_file = os.path.join(tmpdir, "test.pdb")
        with open(pdb_file, "w", encoding="utf-8") as f:
            f.write("ATOM ...\n")
            f.write("HETATM ...\n")
            f.write("OTHER ...\n")
        generic_utils.clean_pdb(pdb_file)
        with open(pdb_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert lines[0].startswith("ATOM")
            assert lines[1].startswith("HETATM")

def test_calculate_averages():
    """Test the calculate_averages function."""
    stats = {1: {"a": 1, "b": 2}, 2: {"a": 3, "b": 4}}
    averages = generic_utils.calculate_averages(stats)
    assert averages["a"] == 2.0
    assert averages["b"] == 3.0

def test_check_filters():
    """Test the check_filters function."""
    data = [1, 2, 3]
    labels = ["a", "b", "c"]
    filters = {"a": {"threshold": 0, "higher": True}}
    assert generic_utils.check_filters(data, labels, filters)
    filters = {"a": {"threshold": 2, "higher": True}}
    assert isinstance(generic_utils.check_filters(data, labels, filters), list)
