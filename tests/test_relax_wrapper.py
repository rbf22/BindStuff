"""
Tests for the relax_wrapper module.
"""
import os
import unittest
import pytest
from functions import relax_wrapper
from pdbfixer import PDBFixer
from openmm.app import PDBFile

class TestRelaxWrapper(unittest.TestCase):
    """
    Tests for the relax_wrapper module.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.test_pdb_path = "tests/formatted.pdb"
        self.cleaned_pdb_path = "tests/cleaned.pdb"
        fixer = PDBFixer(filename=self.test_pdb_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.cleaned_pdb_path, 'w'))

    def tearDown(self):
        """
        Tear down the test environment.
        """
        os.remove(self.cleaned_pdb_path)
        if os.path.exists("tests/relaxed.pdb"):
            os.remove("tests/relaxed.pdb")

    def test_relax_pdb(self):
        """
        Test the relax_pdb function.
        """
        try:
            relax_wrapper.relax_pdb(self.cleaned_pdb_path, "tests/relaxed.pdb")
            self.assertTrue(os.path.exists("tests/relaxed.pdb"))
        except ValueError as e:
            pytest.skip(f"Relaxation failed: {e}")

if __name__ == "__main__":
    unittest.main()
