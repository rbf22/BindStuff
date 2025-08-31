"""
This module provides an adapter to the open-source protein analysis tools,
with function signatures matching the original PyRosetta-based implementation.
"""
from . import scoring
from . import struct_utils
from . import relax_wrapper

def score_interface(pdb_file, binder_chain="B"):
    """
    Analyzes the interface between two chains in a PDB file.
    """
    return scoring.score_interface(pdb_file, binder_chain=binder_chain)


def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Aligns a PDB file to a reference PDB file.
    """
    struct_utils.superimpose_pdbs(
        ref_pdb=reference_pdb,
        mov_pdb=align_pdb,
        ref_chain_id=reference_chain_id,
        mov_chain_id=align_chain_id,
        output_pdb=align_pdb,
    )
    clean_pdb(align_pdb)


def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Calculates the RMSD between two PDB files without alignment.
    """
    return float(struct_utils.compute_unaligned_rmsd(
        ref_pdb=reference_pdb,
        mov_pdb=align_pdb,
        ref_chain_id=reference_chain_id,
        mov_chain_id=align_chain_id,
    ))


def pr_relax(pdb_file, relaxed_pdb_path, allow_bb=True, allow_sc=True, max_iter=200):
    """
    Relaxes a PDB file using the AlphaFold relaxation protocol.
    """
    # Relaxation step
    relax_wrapper.relax_pdb(
        pdb_file=pdb_file,
        output_pdb_path=relaxed_pdb_path,
        max_iterations=max_iter,
    )

    # B-factor copying and cleaning
    copy_bfactors(pdb_file, relaxed_pdb_path)
    clean_pdb(relaxed_pdb_path)


def copy_bfactors(source_pdb, dest_pdb):
    """
    Copies B-factors from a source PDB to a destination PDB.
    """
    source_struct = struct_utils.load_structure(source_pdb)
    dest_struct = struct_utils.load_structure(dest_pdb)

    source_bfactors = {
        (atom.get_parent().id, atom.id): atom.bfactor
        for atom in source_struct.get_atoms()
    }

    for atom in dest_struct.get_atoms():
        key = (atom.get_parent().id, atom.id)
        if key in source_bfactors:
            atom.bfactor = source_bfactors[key]

    io = struct_utils.PDB.PDBIO()
    io.set_structure(dest_struct)
    io.save(dest_pdb)


def clean_pdb(pdb_file):
    """
    Cleans a PDB file by removing all lines that do not start with ATOM, HETATM, TER, or END.
    """
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    cleaned_lines = [
        line for line in lines if line.startswith(('ATOM', 'HETATM', 'TER', 'END'))
    ]

    with open(pdb_file, 'w') as f:
        f.writelines(cleaned_lines)
