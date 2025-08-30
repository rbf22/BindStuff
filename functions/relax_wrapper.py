"""
A wrapper for the AlphaFold relaxation protocol.
This module provides a simple interface to the AmberRelaxation class
from the alphafold.relax package.
"""
from alphafold.common import protein
from alphafold.relax.relax import AmberRelaxation


def relax_pdb(
    pdb_file: str,
    output_pdb_path: str,
    stiffness: float = 1.0,
    max_iterations: int = 200,
    max_outer_iterations: int = 3,
    use_gpu: bool = False,
):
    """
    Runs the Amber relaxation protocol on a PDB file.

    Args:
        pdb_file: The path to the PDB file to relax.
        output_pdb_path: The path to save the relaxed PDB file.
        stiffness: The stiffness of the restraints, in kcal/mol A**2.
        max_iterations: The maximum number of minimization iterations.
        max_outer_iterations: The maximum number of outer iterations.
        use_gpu: Whether to use the GPU for the relaxation.
    """
    with open(pdb_file, "r") as f:
        pdb_string = f.read()

    prot = protein.from_pdb_string(pdb_string)

    relaxer = AmberRelaxation(
        max_iterations=max_iterations,
        tolerance=2.39,  # kcal/mol, same as in the alphafold relax script
        stiffness=stiffness,
        exclude_residues=[],
        max_outer_iterations=max_outer_iterations,
        use_gpu=use_gpu,
    )

    relaxed_pdb_str, _, _ = relaxer.process(prot=prot)

    with open(output_pdb_path, "w") as f:
        f.write(relaxed_pdb_str)
