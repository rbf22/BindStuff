
"""
Structural biology utilities using Biopython.
This module provides functions for protein structure analysis, including
alignment, RMSD calculation, and basic selections.
"""
import warnings
from typing import List, Tuple

import numpy as np
from Bio import PDB, pairwise2
from Bio.PDB import Superimposer
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.Polypeptide import is_aa

warnings.filterwarnings("ignore")

# ---------------------------- Core IO ----------------------------

def load_structure(pdb_file: str) -> PDB.Structure.Structure:
    """Loads a PDB file into a Biopython Structure object."""
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure('protein', pdb_file)

# ------------------------- Selection utils -----------------------

def get_chain_residues(structure: PDB.Structure.Structure, chain_id: str) -> List[PDB.Residue.Residue]:
    """Extracts all standard amino acid residues from a specified chain."""
    return [
        residue
        for model in structure
        if chain_id in [c.id for c in model]
        for residue in model[chain_id]
        if is_aa(residue, standard=True)
    ]

def get_chain_atoms(structure: PDB.Structure.Structure, chain_id: str, heavy_only: bool = True) -> List[PDB.Atom.Atom]:
    """Extracts atoms from a specified chain, optionally excluding hydrogens."""
    atoms = []
    for residue in get_chain_residues(structure, chain_id):
        for atom in residue:
            if heavy_only and atom.element == 'H':
                continue
            atoms.append(atom)
    return atoms

def get_ca_atoms(structure: PDB.Structure.Structure, chain_id: str) -> List[PDB.Atom.Atom]:
    """Extracts all C-alpha atoms from a specified chain."""
    return [
        residue['CA']
        for residue in get_chain_residues(structure, chain_id)
        if 'CA' in residue
    ]

# ------------------------ Alignment/RMSD -------------------------

def map_residues_by_sequence(
    ref_residues: List[PDB.Residue.Residue],
    mov_residues: List[PDB.Residue.Residue]
) -> Tuple[List[PDB.Residue.Residue], List[PDB.Residue.Residue]]:
    """
    Aligns two lists of residues by sequence and returns the corresponding pairs.
    """
    ref_seq = "".join([protein_letters_3to1.get(res.resname, 'X') for res in ref_residues])
    mov_seq = "".join([protein_letters_3to1.get(res.resname, 'X') for res in mov_residues])

    alignments = pairwise2.align.globalxx(ref_seq, mov_seq)
    if not alignments:
        return [], []

    alignment = alignments[0]
    aligned_ref_seq, aligned_mov_seq = alignment[0], alignment[1]

    aligned_ref_residues = []
    aligned_mov_residues = []
    ref_idx, mov_idx = 0, 0

    for i in range(len(aligned_ref_seq)):
        if aligned_ref_seq[i] != '-' and aligned_mov_seq[i] != '-':
            aligned_ref_residues.append(ref_residues[ref_idx])
            aligned_mov_residues.append(mov_residues[mov_idx])
        if aligned_ref_seq[i] != '-':
            ref_idx += 1
        if aligned_mov_seq[i] != '-':
            mov_idx += 1

    return aligned_ref_residues, aligned_mov_residues


def superimpose_pdbs(
    ref_pdb: str,
    mov_pdb: str,
    ref_chain_id: str,
    mov_chain_id: str,
    output_pdb: str
):
    """
    Superimposes the mobile PDB onto the reference PDB based on C-alpha alignment
    of specified chains and saves the transformed mobile structure.
    """
    ref_struct = load_structure(ref_pdb)
    mov_struct = load_structure(mov_pdb)

    ref_residues = get_chain_residues(ref_struct, ref_chain_id)
    mov_residues = get_chain_residues(mov_struct, mov_chain_id)

    aligned_ref_residues, aligned_mov_residues = map_residues_by_sequence(
        ref_residues, mov_residues
    )

    ref_atoms = [res['CA'] for res in aligned_ref_residues if 'CA' in res]
    mov_atoms = [res['CA'] for res in aligned_mov_residues if 'CA' in res]

    if len(ref_atoms) != len(mov_atoms):
        raise ValueError("Cannot superimpose: C-alpha atoms count mismatch after alignment.")

    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, mov_atoms)
    superimposer.apply(mov_struct.get_atoms())

    io = PDB.PDBIO()
    io.set_structure(mov_struct)
    io.save(output_pdb)


def compute_unaligned_rmsd(
    ref_pdb: str,
    mov_pdb: str,
    ref_chain_id: str,
    mov_chain_id: str
) -> float:
    """
    Computes the RMSD between two PDB files without alignment based on C-alpha
    atoms of specified chains.
    """
    ref_struct = load_structure(ref_pdb)
    mov_struct = load_structure(mov_pdb)

    ref_residues = get_chain_residues(ref_struct, ref_chain_id)
    mov_residues = get_chain_residues(mov_struct, mov_chain_id)

    aligned_ref_residues, aligned_mov_residues = map_residues_by_sequence(
        ref_residues, mov_residues
    )

    ref_atoms = [res['CA'] for res in aligned_ref_residues if 'CA' in res]
    mov_atoms = [res['CA'] for res in aligned_mov_residues if 'CA' in res]

    if len(ref_atoms) != len(mov_atoms):
        raise ValueError("Cannot compute RMSD: C-alpha atoms count mismatch after alignment.")

    # Manual RMSD calculation
    diff = np.array([ref_atoms[i].coord - mov_atoms[i].coord for i in range(len(ref_atoms))])
    rmsd = np.sqrt(np.sum(diff**2) / len(ref_atoms))

    return float(round(rmsd, 2))
