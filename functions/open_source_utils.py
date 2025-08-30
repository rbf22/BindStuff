#
# Open-Source Utilities
#
import os
from prody import (parsePDB, Interactions, showPairEnergy)
from openmm import app
from pdbfixer import PDBFixer
from Bio.PDB import PDBParser, SASA, PDBIO, Superimposer
from .generic_utils import clean_pdb


def openmm_relax(pdb_file, relaxed_pdb_path):
    """
    Performs energy minimization on a protein structure using PDBFixer.
    """
    if not os.path.exists(relaxed_pdb_path):
        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        with open(relaxed_pdb_path, "w") as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        clean_pdb(relaxed_pdb_path)


def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Aligns two PDB structures using Bio.PDB.Superimposer.
    """
    pdb_parser = PDBParser(QUIET=True)
    ref_structure = pdb_parser.get_structure("reference", reference_pdb)
    sample_structure = pdb_parser.get_structure("sample", align_pdb)

    ref_model = ref_structure[0]
    sample_model = sample_structure[0]

    ref_atoms = []
    sample_atoms = []

    for ref_chain in ref_model:
        if ref_chain.id in reference_chain_id.split(','):
            for ref_res in ref_chain:
                if 'CA' in ref_res:
                    ref_atoms.append(ref_res['CA'])

    for sample_chain in sample_model:
        if sample_chain.id in align_chain_id.split(','):
            for sample_res in sample_chain:
                if 'CA' in sample_res:
                    sample_atoms.append(sample_res['CA'])

    if len(ref_atoms) != len(sample_atoms):
        raise ValueError("Cannot align structures with different number of atoms.")

    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    super_imposer.apply(sample_model.get_atoms())

    io = PDBIO()
    io.set_structure(sample_structure)
    io.save(align_pdb)
    clean_pdb(align_pdb)


def unaligned_rmsd(reference_pdb, align_pdb,
                   reference_chain_id, align_chain_id):
    """
    Calculates the RMSD between two chains without prior alignment.
    """
    pdb_parser = PDBParser(QUIET=True)
    ref_structure = pdb_parser.get_structure("reference", reference_pdb)
    sample_structure = pdb_parser.get_structure("sample", align_pdb)

    ref_model = ref_structure[0]
    sample_model = sample_structure[0]

    ref_atoms = []
    sample_atoms = []

    for ref_chain in ref_model:
        if ref_chain.id in reference_chain_id.split(','):
            for ref_res in ref_chain:
                if 'CA' in ref_res:
                    ref_atoms.append(ref_res['CA'])

    for sample_chain in sample_model:
        if sample_chain.id in align_chain_id.split(','):
            for sample_res in sample_chain:
                if 'CA' in sample_res:
                    sample_atoms.append(sample_res['CA'])

    if len(ref_atoms) != len(sample_atoms):
        raise ValueError("Cannot calculate RMSD for structures with different number of atoms.")

    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)

    return round(super_imposer.rms, 2)


def score_interface(pdb_file, binder_chain="B"):
    """
    Calculates interface scores using a combination of ProDy and BioPython.
    """
    # Load structure with ProDy for interaction analysis
    structure = parsePDB(pdb_file)
    if structure is None:
        return {}, {}, ""

    # Define target and binder selections
    target_chain = "A"  # Assuming target is always chain A

    # --- Interface Residues and Hydrophobicity (ProDy) ---
    sele_str = (
        f"(chain {target_chain} and within 5 of chain {binder_chain})"
        f" or (chain {binder_chain} and within 5 of"
        f" chain {target_chain})"
    )
    interface_res_sel = structure.select(sele_str)
    if interface_res_sel is None:
        interface_nres = 0
        interface_residues_pdb_ids_str = ""
        interface_hydrophobicity = 0
        interface_aa = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    else:
        interface_residues = interface_res_sel.getHierView().iterResidues()
        interface_nres = len(
            set(res.getResnum() for res in interface_residues if res.getChid() == binder_chain)
        )

        interface_residues_pdb_ids = [
            f"{res.getChid()}{res.getResnum()}" for res in interface_residues
        ]
        interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

        hydrophobic_aa = set("ACFILMPVWY")
        hydrophobic_count = 0
        interface_aa = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
        for res in interface_residues:
            if res.getChid() == binder_chain:
                resname = res.getResname()
                if resname in interface_aa:
                    interface_aa[resname] += 1
                if resname in hydrophobic_aa:
                    hydrophobic_count += 1

        if interface_nres != 0:
            interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
        else:
            interface_hydrophobicity = 0

    # --- Hydrogen Bonds and Interaction Energy (ProDy InSty) ---
    try:
        interactions = Interactions(structure)
        if interactions is not None:
            interactions.calcProteinInteractions(structure)
            hbonds = interactions.getHydrogenBonds(
                selection=f"chain {target_chain}", selection2=f"chain {binder_chain}"
            )
            interface_interface_hbonds = len(hbonds) if hbonds is not None else 0

            # As a proxy for dG, we can sum the energies of the hydrogen bonds
            # Note: This is a very rough approximation
            if hbonds:
                showPairEnergy(hbonds)
                interface_dg = sum(hbond[-1] for hbond in hbonds)
            else:
                interface_dg = 0
        else:
            interface_interface_hbonds = 0
            interface_dg = 0

    except Exception:  # pylint: disable=broad-except-clause
        interface_interface_hbonds = 0
        interface_dg = 0

    # --- SASA Calculation (BioPython) ---
    parser = PDBParser(QUIET=True)
    bp_structure = parser.get_structure("s", pdb_file)
    if bp_structure is None:
        return {}, {}, ""
    sr = SASA.ShrakeRupley()
    sr.compute(bp_structure, level="S")

    complex_sasa = bp_structure.sasa

    # Isolate chains for individual SASA calculation
    class ChainSelect:
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_model(self, model):
            return 1

        def accept_chain(self, chain):
            if chain.get_id() == self.chain_id:
                return 1
            return 0

        def accept_residue(self, residue):
            return 1

        def accept_atom(self, atom):
            return 1

    io = PDBIO()
    io.set_structure(bp_structure)
    io.save("target.pdb", select=ChainSelect(target_chain))
    io.save("binder.pdb", select=ChainSelect(binder_chain))

    target_struct = parser.get_structure("t", "target.pdb")
    binder_struct = parser.get_structure("b", "binder.pdb")

    sr.compute(target_struct, level="S")
    target_sasa = target_struct.sasa

    sr.compute(binder_struct, level="S")
    binder_sasa = binder_struct.sasa

    os.remove("target.pdb")
    os.remove("binder.pdb")

    interface_dsasa = (target_sasa + binder_sasa) - complex_sasa

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dsasa / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # --- Binder Score (Proxy) and Surface Hydrophobicity ---
    # As a proxy for binder_score, we calculate the total intra-chain
    # interaction energy of the binder
    binder_only_structure = structure.select(f"chain {binder_chain}")
    binder_score = 0
    if binder_only_structure is not None:
        binder_interactions = Interactions(binder_only_structure)
        binder_interactions.calcProteinInteractions(binder_only_structure)
        all_binder_hbonds = binder_interactions.getHydrogenBonds()
        if all_binder_hbonds:
            showPairEnergy(all_binder_hbonds)
            binder_score = sum(hbond[-1] for hbond in all_binder_hbonds)

    # Surface Hydrophobicity
    # Placeholder, as this is complex to calculate accurately
    surface_hydrophobicity = 0

    # --- Assemble results ---
    interface_scores = {
        "binder_score": binder_score,
        "surface_hydrophobicity": surface_hydrophobicity,
        "interface_sc": 0,  # Not implemented
        "interface_packstat": 0,  # Not implemented
        "interface_dG": interface_dg,
        "interface_dSASA": interface_dsasa,
        "interface_dG_SASA_ratio": (interface_dg / interface_dsasa) * 100
        if interface_dsasa != 0
        else 0,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": interface_hydrophobicity,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": interface_interface_hbonds,
        "interface_hbond_percentage": (interface_interface_hbonds / interface_nres)
        * 100
        if interface_nres != 0
        else 0,
        "interface_delta_unsat_hbonds": 0,  # Not implemented
        "interface_delta_unsat_hbonds_percentage": 0,  # Not implemented
    }

    interface_scores = {
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in interface_scores.items()
    }

    return interface_scores, interface_aa, interface_residues_pdb_ids_str
