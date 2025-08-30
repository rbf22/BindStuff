"""
Tests for the scoring module.
"""
import unittest
from functions import scoring

class TestScoring(unittest.TestCase):
    """
    Tests for the scoring module.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.test_pdb_path = "tests/test_complex.pdb"

    def test_score_interface(self):
        """
        Test the score_interface function.
        """
        scores, _, _ = scoring.score_interface(self.test_pdb_path, binder_chain="B")
        self.assertIsInstance(scores, dict)
        self.assertIn("interface_nres", scores)

if __name__ == "__main__":
    unittest.main()
