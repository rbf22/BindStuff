"""Tests for the generic_utils module."""
import os
import json
import tempfile
import zipfile
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
            json.dump({"1_pLDDT": {"threshold": 0.7}, "InterfaceAAs": {}}, f)
        generic_utils.generate_filter_pass_csv(failure_csv, filter_json)
        assert os.path.exists(failure_csv)
        df = pd.read_csv(failure_csv)
        assert "pLDDT" in df.columns
        assert "InterfaceAAs_A" in df.columns

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
        generic_utils.update_failures(failure_csv, "1_col3")
        df = pd.read_csv(failure_csv)
        assert df["col3"][0] == 1
        generic_utils.update_failures(failure_csv, {"2_col4": 3})
        df = pd.read_csv(failure_csv)
        assert df["col4"][0] == 3


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
        advanced_settings = {"max_trajectories": False}
        assert not generic_utils.check_n_trajectories(design_paths, advanced_settings)

def test_check_accepted_designs():
    """Test the check_accepted_designs function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        design_paths = generic_utils.generate_directories(tmpdir)
        mpnn_csv = os.path.join(tmpdir, "mpnn.csv")
        final_csv = os.path.join(tmpdir, "final.csv")
        _, design_labels, final_labels = generic_utils.generate_dataframe_labels()

        # Create dummy files and data
        pd.DataFrame(columns=design_labels).to_csv(mpnn_csv, index=False)
        pd.DataFrame(columns=final_labels).to_csv(final_csv, index=False)

        advanced_settings = {"zip_animations": True, "zip_plots": True}
        target_settings = {"number_of_final_designs": 1, "binder_name": "test_binder"}

        # Case 1: Not enough accepted designs
        assert not generic_utils.check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels)

        # Create an accepted design
        open(os.path.join(design_paths["Accepted"], "test_binder_model1.pdb"), 'w').close()

        # Add data to mpnn_csv
        mpnn_df = pd.DataFrame([{"Design": "test_binder", "Average_i_pTM": 0.8}], columns=design_labels)
        mpnn_df.to_csv(mpnn_csv, index=False)

        # Case 2: Enough accepted designs
        assert generic_utils.check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels)

        # Verify ranked file exists
        assert os.path.exists(os.path.join(design_paths["Accepted/Ranked"], "1_test_binder_model1.pdb"))

def test_load_helicity():
    """Test the load_helicity function."""
    adv_set = {"random_helicity": False, "weights_helicity": 0.5}
    assert generic_utils.load_helicity(adv_set) == 0.5
    adv_set = {"random_helicity": True, "weights_helicity": 0}
    assert isinstance(generic_utils.load_helicity(adv_set), float)
    adv_set = {"random_helicity": False, "weights_helicity": 0}
    assert generic_utils.load_helicity(adv_set) == 0

@patch("jax.devices")
def test_check_jax_gpu(mock_devices):
    """Test the check_jax_gpu function."""
    mock_devices.return_value = [MagicMock(platform="gpu")]
    with patch("builtins.print") as mock_print:
        generic_utils.check_jax_gpu()
        mock_print.assert_any_call("Available GPUs:")

    mock_devices.return_value = [MagicMock(platform="cpu")]
    with patch('sys.exit') as mock_exit:
        generic_utils.check_jax_gpu()
        mock_exit.assert_called_once_with(1)


def test_perform_input_check():
    """Test the perform_input_check function."""
    args = MagicMock()
    args.settings = "settings.json"
    args.filters = None
    args.advanced = None
    _, f, a = generic_utils.perform_input_check(args)
    assert f is not None
    assert a is not None

    with patch('sys.exit') as mock_exit:
        args.settings = None
        generic_utils.perform_input_check(args)
        mock_exit.assert_called_once_with(1)

def test_perform_advanced_settings_check():
    """Test the perform_advanced_settings_check function."""
    adv_set = {
        "af_params_dir": None, "dssp_path": None,
        "dalphaball_path": None, "omit_AAs": " C "
    }
    adv_set_colab = generic_utils.perform_advanced_settings_check(adv_set, "colab")
    assert adv_set_colab["af_params_dir"]

    adv_set = {
        "af_params_dir": "", "dssp_path": "",
        "dalphaball_path": "", "omit_AAs": False
    }

    adv_set_local = generic_utils.perform_advanced_settings_check(adv_set, "test_folder")
    assert adv_set_local["af_params_dir"] == os.path.join("test_folder", "params")
    assert adv_set_local["omit_AAs"] is None

def test_load_json_settings():
    """Test load_json_settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_file = os.path.join(tmpdir, "settings.json")
        filters_file = os.path.join(tmpdir, "filters.json")
        advanced_file = os.path.join(tmpdir, "advanced.json")

        with open(settings_file, "w") as f:
            json.dump({"a": 1}, f)
        with open(filters_file, "w") as f:
            json.dump({"b": 2}, f)
        with open(advanced_file, "w") as f:
            json.dump({"c": 3}, f)

        s, a, f = generic_utils.load_json_settings(settings_file, filters_file, advanced_file)
        assert s == {"a": 1}
        assert f == {"b": 2}
        assert a == {"c": 3}

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

def test_zip_and_empty_folder():
    """Test the zip_and_empty_folder function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder_to_zip = os.path.join(tmpdir, "test_folder")
        os.makedirs(folder_to_zip)
        file_to_zip = os.path.join(folder_to_zip, "test.txt")
        with open(file_to_zip, "w") as f:
            f.write("test")

        generic_utils.zip_and_empty_folder(folder_to_zip, ".txt")

        zip_path = os.path.join(tmpdir, "test_folder.zip")
        assert os.path.exists(zip_path)
        assert not os.path.exists(file_to_zip)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            assert "test.txt" in zf.namelist()


def test_calculate_averages():
    """Test the calculate_averages function."""
    stats = {1: {"a": 1, "b": 2, "InterfaceAAs": {"A": 1}}, 2: {"a": 3, "b": 4, "InterfaceAAs": {"C": 1}}}
    averages = generic_utils.calculate_averages(stats, handle_aa=True)
    assert averages["a"] == 2.0
    assert averages["b"] == 3.0
    assert averages["InterfaceAAs"] > 0

    stats_with_none = {1: {"a": 1, "b": None}, 2: {"a": 3, "b": 4}}
    averages = generic_utils.calculate_averages(stats_with_none)
    assert averages['b'] == 2.0


def test_check_filters():
    """Test the check_filters function."""
    data = [1, 2, 3, {"A": 1}]
    labels = ["a", "b", "c", "Average_InterfaceAAs"]
    filters = {"a": {"threshold": 0, "higher": True}}
    assert generic_utils.check_filters(data, labels, filters)

    filters = {"a": {"threshold": 2, "higher": True}}
    assert "a" in generic_utils.check_filters(data, labels, filters)

    filters = {"a": {"threshold": 0, "higher": False}}
    assert "a" in generic_utils.check_filters(data, labels, filters)

    filters = {"d": {"threshold": 0, "higher": True}} # label not in data
    assert generic_utils.check_filters(data, labels, filters)

    filters = {"Average_InterfaceAAs": {"A": {"threshold": 0, "higher": True}}}
    assert generic_utils.check_filters(data, labels, filters)

    filters = {"Average_InterfaceAAs": {"A": {"threshold": 2, "higher": True}}}
    assert "Average_InterfaceAAs_A" in generic_utils.check_filters(data, labels, filters)

    filters = {"Average_InterfaceAAs": {"A": {"threshold": 0, "higher": False}}}
    assert "Average_InterfaceAAs_A" in generic_utils.check_filters(data, labels, filters)
