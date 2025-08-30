"""
Utility functions for open source protein structure analysis.
This module is now a wrapper around the open_source_adapter.
"""
from . import open_source_adapter


def openmm_relax(pdb_file: str, relaxed_pdb_path: str) -> None:
    """
    Relaxes a PDB file using the open source implementation.
    """
    open_source_adapter.pr_relax(pdb_file, relaxed_pdb_path)


def align_pdbs(reference_pdb: str, align_pdb: str, reference_chain_id: str, align_chain_id: str) -> None:
    """
    Aligns two PDB structures using the open source implementation.
    """
    open_source_adapter.align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id)


def unaligned_rmsd(reference_pdb: str, align_pdb: str,
                   reference_chain_id: str, align_chain_id: str) -> float:
    """
    Calculates the RMSD between two chains without prior alignment using the open source implementation.
    """
    return open_source_adapter.unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id)


def score_interface(pdb_file: str, binder_chain: str = "B"):
    """
    Calculates interface scores using the open source implementation.
    """
    return open_source_adapter.score_interface(pdb_file, binder_chain)
