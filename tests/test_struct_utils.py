"""
Tests for the struct_utils module.
"""
import os
import unittest
import shutil
from functions import struct_utils

class TestStructUtils(unittest.TestCase):
    """
    Tests for the struct_utils module.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.test_pdb_path = "tests/test_complex.pdb"
        self.align_pdb_path = "tests/align.pdb"
        shutil.copy(self.test_pdb_path, self.align_pdb_path)

    def tearDown(self):
        """
        Tear down the test environment.
        """
        os.remove(self.align_pdb_path)
        if os.path.exists("tests/aligned.pdb"):
            os.remove("tests/aligned.pdb")

    def test_load_structure(self):
        """
        Test the load_structure function.
        """
        structure = struct_utils.load_structure(self.test_pdb_path)
        self.assertIsNotNone(structure)

    def test_get_chain_residues(self):
        """
        Test the get_chain_residues function.
        """
        structure = struct_utils.load_structure(self.test_pdb_path)
        residues_a = struct_utils.get_chain_residues(structure, "A")
        residues_b = struct_utils.get_chain_residues(structure, "B")
        self.assertEqual(len(residues_a), 2)
        self.assertEqual(len(residues_b), 2)

    def test_superimpose_pdbs(self):
        """
        Test the superimpose_pdbs function.
        """
        struct_utils.superimpose_pdbs(self.test_pdb_path, self.align_pdb_path, "A", "B", "tests/aligned.pdb")
        self.assertTrue(os.path.exists("tests/aligned.pdb"))

    def test_compute_unaligned_rmsd(self):
        """
        Test the compute_unaligned_rmsd function.
        """
        rmsd = struct_utils.compute_unaligned_rmsd(self.test_pdb_path, self.align_pdb_path, "A", "B")
        self.assertIsInstance(rmsd, float)

if __name__ == "__main__":
    unittest.main()
