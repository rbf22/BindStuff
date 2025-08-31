"""
Tests for the struct_utils module.
"""
import shutil
import pytest
from functions import struct_utils


@pytest.fixture
def pdb_files(tmp_path):
    """
    Fixture to set up test PDBs for struct_utils.
    Returns paths to test_pdb and align_pdb.
    """
    test_pdb_path = "tests/test_complex.pdb"
    align_pdb_path = tmp_path / "align.pdb"

    shutil.copy(test_pdb_path, align_pdb_path)

    return test_pdb_path, align_pdb_path


def test_load_structure(pdb_files):
    """
    Test the load_structure function.
    """
    test_pdb, _ = pdb_files
    structure = struct_utils.load_structure(test_pdb)
    assert structure is not None


def test_get_chain_residues(pdb_files):
    """
    Test the get_chain_residues function.
    """
    test_pdb, _ = pdb_files
    structure = struct_utils.load_structure(test_pdb)
    residues_a = struct_utils.get_chain_residues(structure, "A")
    residues_b = struct_utils.get_chain_residues(structure, "B")

    # Expected residue counts (matches original unittest)
    assert len(residues_a) == 2
    assert len(residues_b) == 2


def test_superimpose_pdbs(pdb_files, tmp_path):
    """
    Test the superimpose_pdbs function.
    """
    test_pdb, align_pdb = pdb_files
    aligned_path = tmp_path / "aligned.pdb"

    struct_utils.superimpose_pdbs(test_pdb, align_pdb, "A", "B", str(aligned_path))
    assert aligned_path.exists()


def test_compute_unaligned_rmsd(pdb_files):
    """
    Test the compute_unaligned_rmsd function.
    """
    test_pdb, align_pdb = pdb_files
    rmsd = struct_utils.compute_unaligned_rmsd(test_pdb, align_pdb, "A", "B")
    assert isinstance(rmsd, float)