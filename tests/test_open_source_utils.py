"""
Tests for the open_source_utils module.
"""
import os
import shutil
import pytest
from functions import open_source_utils
from pdbfixer import PDBFixer
from openmm.app import PDBFile


@pytest.fixture
def pdb_files(tmp_path):
    """
    Fixture to prepare input PDBs for testing.
    Returns paths to complex_pdb, align_pdb, and cleaned_pdb.
    """
    complex_pdb_path = "tests/test_complex.pdb"
    align_pdb_path = tmp_path / "align.pdb"
    cleaned_pdb_path = tmp_path / "cleaned.pdb"

    # Copy complex_pdb -> align_pdb
    shutil.copy(complex_pdb_path, align_pdb_path)

    # Clean the complex PDB
    fixer = PDBFixer(filename=complex_pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(cleaned_pdb_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return complex_pdb_path, align_pdb_path, cleaned_pdb_path


def test_open_source_utils(pdb_files, tmp_path):
    """
    Test the open_source_utils functions.
    """
    complex_pdb, align_pdb, cleaned_pdb = pdb_files
    relaxed_path = tmp_path / "relaxed.pdb"

    # Test align_pdbs
    open_source_utils.align_pdbs(str(complex_pdb), str(align_pdb), "A", "B")

    # Test unaligned_rmsd
    rmsd = open_source_utils.unaligned_rmsd(complex_pdb, align_pdb, "A", "B")
    assert isinstance(rmsd, float)

    # Test score_interface
    scores, _, _ = open_source_utils.score_interface(complex_pdb, binder_chain="B")
    assert isinstance(scores, dict)

    # Test openmm_relax
    open_source_utils.openmm_relax(str(cleaned_pdb), str(relaxed_path))
    assert relaxed_path.exists()