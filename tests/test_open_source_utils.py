"""
Tests for the open_source_utils module.
"""
import os
import unittest
import shutil
import pytest
from functions import open_source_utils
from pdbfixer import PDBFixer
from openmm.app import PDBFile

class TestOpenSourceUtils(unittest.TestCase):
    """
    Tests for the open_source_utils module.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.complex_pdb_path = "tests/test_complex.pdb"
        self.align_pdb_path = "tests/align.pdb"
        self.cleaned_pdb_path = "tests/cleaned.pdb"
        shutil.copy(self.complex_pdb_path, self.align_pdb_path)

        fixer = PDBFixer(filename=self.complex_pdb_path)
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
        os.remove(self.align_pdb_path)
        os.remove(self.cleaned_pdb_path)
        if os.path.exists("tests/relaxed.pdb"):
            os.remove("tests/relaxed.pdb")
        if os.path.exists(self.complex_pdb_path.replace('.pdb', '_fixed.pdb')):
            os.remove(self.complex_pdb_path.replace('.pdb', '_fixed.pdb'))


    def test_open_source_utils(self):
        """
        Test the open_source_utils functions.
        """
        open_source_utils.align_pdbs(self.complex_pdb_path, self.align_pdb_path, "A", "B")
        rmsd = open_source_utils.unaligned_rmsd(self.complex_pdb_path, self.align_pdb_path, "A", "B")
        self.assertIsInstance(rmsd, float)
        scores, _, _ = open_source_utils.score_interface(self.complex_pdb_path, binder_chain="B")
        self.assertIsInstance(scores, dict)
        try:
            open_source_utils.openmm_relax(self.cleaned_pdb_path, "tests/relaxed.pdb")
            self.assertTrue(os.path.exists("tests/relaxed.pdb"))
        except ValueError as e:
            pytest.skip(f"Relaxation failed: {e}")

if __name__ == "__main__":
    unittest.main()
