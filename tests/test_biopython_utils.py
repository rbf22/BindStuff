"""Tests for the biopython_utils module."""
from unittest.mock import patch, MagicMock

import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom
import pytest

from functions import biopython_utils


@pytest.fixture
def mock_structure():
    """Create a mock structure for testing."""
    # Create a mock atom
    mock_atom_ca = Atom.Atom("CA", np.array([0, 0, 0]), 20.0, 1.0, " ", " CA ", 0, "C")
    mock_atom_cb = Atom.Atom("CB", np.array([1, 1, 1]), 20.0, 1.0, " ", " CB ", 0, "C")
    # Create a mock residue
    mock_residue = Residue.Residue((" ", 1, " "), "ALA", " ")
    mock_residue.add(mock_atom_ca)
    mock_residue.add(mock_atom_cb)
    # Create a mock chain
    mock_chain_a = Chain.Chain("A")
    mock_chain_a.add(mock_residue)
    mock_chain_b = Chain.Chain("B")
    mock_chain_b.add(mock_residue)
    # Create a mock model
    mock_model = Model.Model(0)
    mock_model.add(mock_chain_a)
    mock_model.add(mock_chain_b)
    # Create a mock structure
    structure = Structure.Structure("test")
    structure.add(mock_model)
    return structure

@patch('functions.biopython_utils.ProteinAnalysis')
def test_validate_design_sequence(mock_protein_analysis):
    """Test the validate_design_sequence function."""
    # Mock ProteinAnalysis
    mock_analysis_instance = MagicMock()
    mock_analysis_instance.molar_extinction_coefficient.return_value = (1000, 0)
    mock_analysis_instance.molecular_weight.return_value = 1000
    mock_protein_analysis.return_value = mock_analysis_instance

    # Test case 1: Clashes
    notes = biopython_utils.validate_design_sequence("ACGT", 1, {"omit_AAs": ""})
    assert "Relaxed structure contains clashes." in notes

    # Test case 2: Restricted AAs
    notes = biopython_utils.validate_design_sequence("ACGT", 0, {"omit_AAs": "C"})
    assert "Contains: C!" in notes

    # Test case 3: Low absorption
    mock_analysis_instance.molar_extinction_coefficient.return_value = (10, 0)
    notes = biopython_utils.validate_design_sequence("ACGT", 0, {"omit_AAs": ""})
    assert "Absorption value is" in notes

    # Test case 4: All good
    mock_analysis_instance.molar_extinction_coefficient.return_value = (10000, 0)
    notes = biopython_utils.validate_design_sequence("ACGT", 0, {"omit_AAs": ""})
    assert notes == ""

@patch('functions.biopython_utils.PDBParser')
def test_target_pdb_rmsd(mock_parser, mock_structure):
    """Test the target_pdb_rmsd function."""
    mock_parser.return_value.get_structure.return_value = mock_structure
    rmsd = biopython_utils.target_pdb_rmsd("traj.pdb", "start.pdb", "B")
    assert isinstance(rmsd, float)

@patch('functions.biopython_utils.PDBParser')
def test_calculate_clash_score(mock_parser, mock_structure):
    """Test the calculate_clash_score function."""
    mock_parser.return_value.get_structure.return_value = mock_structure
    score = biopython_utils.calculate_clash_score("test.pdb")
    assert isinstance(score, int)

@patch('functions.biopython_utils.PDBParser')
def test_hotspot_residues(mock_parser, mock_structure):
    """Test the hotspot_residues function."""
    mock_parser.return_value.get_structure.return_value = mock_structure
    hotspots = biopython_utils.hotspot_residues("test.pdb")
    assert isinstance(hotspots, dict)

@patch('functions.biopython_utils.hotspot_residues')
@patch('functions.biopython_utils.PDBParser')
@patch('functions.biopython_utils.DSSP')
def test_calc_ss_percentage(mock_dssp, mock_parser, mock_hotspot_residues, mock_structure):
    """Test the calc_ss_percentage function."""
    mock_parser.return_value.get_structure.return_value = mock_structure
    mock_dssp.return_value = {
        ('B', 1): ('ALA', 'H', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 0.0, 0.0, 0.0, 0.0)
    }
    mock_hotspot_residues.return_value = {}
    percentages = biopython_utils.calc_ss_percentage("test.pdb", {"dssp_path": ""})
    assert len(percentages) == 8
    assert isinstance(percentages[0], float)

def test_calculate_percentages():
    """Test the calculate_percentages function."""
    h, s, loop = biopython_utils.calculate_percentages(10, 2, 3)
    assert h == 20.0
    assert s == 30.0
    assert loop == 50.0
