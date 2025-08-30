"""
Tests for the relax_wrapper module.
"""
import os
import pytest
from functions import relax_wrapper
from pdbfixer import PDBFixer
from openmm.app import PDBFile

@pytest.fixture
def cleaned_pdb(tmp_path):
    """
    Fixture to create a cleaned PDB file for testing.
    """
    test_pdb_path = "tests/formatted.pdb"
    cleaned_pdb_path = tmp_path / "cleaned.pdb"

    fixer = PDBFixer(filename=test_pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    with open(cleaned_pdb_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return cleaned_pdb_path


def test_relax_pdb(cleaned_pdb, tmp_path):
    relaxed_path = tmp_path / "relaxed.pdb"
    relax_wrapper.relax_pdb(str(cleaned_pdb), str(relaxed_path))
    
    # Copy to a permanent folder after the test
    permanent_path = "tests/relaxed_saved.pdb"
    print("here")
    import shutil
    shutil.copy(relaxed_path, permanent_path)

    assert os.path.exists(permanent_path)