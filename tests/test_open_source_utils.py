"""Tests for open source utility functions."""
import os
import pytest
from openmm.app import PDBFile
from functions.open_source_utils import (openmm_relax,
                                         align_pdbs,
                                         unaligned_rmsd,
                                         score_interface)
from pdbfixer import PDBFixer

@pytest.fixture
def pdb_files():
    """Create temporary PDB files for testing."""
    pdb_content = """
ATOM      1  N   ALA A   1      27.340  36.634  19.236  1.00  0.00           N
ATOM      2  CA  ALA A   1      28.381  35.698  19.324  1.00  0.00           C
ATOM      3  C   ALA A   1      29.438  36.213  20.245  1.00  0.00           C
ATOM      4  O   ALA A   1      29.338  37.208  20.789  1.00  0.00           O
ATOM      5  CB  ALA A   1      28.031  34.586  18.328  1.00  0.00           C
ATOM      6  N   GLY A   2      30.438  35.586  20.435  1.00  0.00           N
ATOM      7  CA  GLY A   2      31.597  35.968  21.212  1.00  0.00           C
ATOM      8  C   GLY A   2      32.628  34.869  21.231  1.00  0.00           C
ATOM      9  O   GLY A   2      32.488  33.963  20.511  1.00  0.00           O
ATOM     10  N   LEU B   1      33.668  34.939  22.011  1.00  0.00           N
ATOM     11  CA  LEU B   1      34.787  34.029  22.069  1.00  0.00           C
ATOM     12  C   LEU B   1      35.808  34.596  22.989  1.00  0.00           C
ATOM     13  O   LEU B   1      35.658  35.580  23.589  1.00  0.00           O
ATOM     14  CB  LEU B   1      35.213  33.015  21.013  1.00  0.00           C
ATOM     15  CG  LEU B   1      36.258  31.989  21.328  1.00  0.00           C
ATOM     16  CD1 LEU B   1      36.689  31.159  20.142  1.00  0.00           C
ATOM     17  CD2 LEU B   1      37.478  32.518  21.996  1.00  0.00           C
"""
    with open("test.pdb", "w", encoding='utf-8') as f:
        f.write(pdb_content)
    with open("test2.pdb", "w", encoding='utf-8') as f:
        f.write(pdb_content)

    yield "test.pdb", "test2.pdb"

    os.remove("test.pdb")
    os.remove("test2.pdb")
    if os.path.exists("test_relaxed.pdb"):
        os.remove("test_relaxed.pdb")

def test_openmm_relax(test_pdb_files):
    """Test OpenMM energy minimization."""
    pdb1, _ = test_pdb_files
    openmm_relax(pdb1, "test_relaxed.pdb")
    assert os.path.exists("test_relaxed.pdb")

def test_align_pdbs(test_pdb_files):
    """Test PDB structure alignment."""
    pdb1, pdb2 = test_pdb_files
    # Introduce a slight modification to pdb2 to make it different from pdb1
    with open(pdb2, "r+", encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(content.replace("21.996", "22.996"))
    align_pdbs(pdb1, pdb2, "A", "A")
    rmsd = unaligned_rmsd(pdb1, pdb2, "A", "A")
    assert rmsd == 0.0

def test_unaligned_rmsd(test_pdb_files):
    """Test unaligned RMSD calculation."""
    pdb1, pdb2 = test_pdb_files
    rmsd = unaligned_rmsd(pdb1, pdb2, "A", "A")
    assert rmsd == 0.0

def test_score_interface(test_pdb_files):
    """Test interface scoring functionality."""
    pdb1, _ = test_pdb_files
    fixer = PDBFixer(filename=pdb1)
    fixer.addMissingHydrogens(7.0)
    with open("test_with_h.pdb", "w", encoding='utf-8') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    scores, _, _ = score_interface("test_with_h.pdb")
    assert isinstance(scores, dict)
