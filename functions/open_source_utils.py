#
# Open-Source Utilities
#
import os
from prody import (parsePDB, superpose, applyTransformation,
                   writePDB, calcRMSD, Interactions, showPairEnergy)
from openmm.app import (PDBFile, ForceField, NoCutoff,
                        HBonds, Simulation)
from openmm import CustomExternalForce, LangevinIntegrator
from openmm import unit
from Bio.PDB import PDBParser, SASA, PDBIO
from .generic_utils import clean_pdb


def openmm_relax(pdb_file, relaxed_pdb_path):
    """
    Performs energy minimization on a protein structure using OpenMM.
    """
    if not os.path.exists(relaxed_pdb_path):
        # Load PDB using OpenMM
        pdb = PDBFile(pdb_file)

        # Load PDB using ProDy for alignment and b-factors
        original_structure = parsePDB(pdb_file)

        # Setup ForceField
        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

        # Create system
        system = forcefield.createSystem(
            pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds
        )

        # Add position restraints to heavy atoms
        restraint = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addPerParticleParameter("k")
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        system.addForce(restraint)
        for atom in pdb.topology.atoms():
            if atom.element.symbol != "H":
                pos = pdb.positions[atom.index]
                restraint.addParticle(
                    atom.index,
                    [
                        10.0 * unit.kilocalories_per_mole / unit.angstrom**2,
                        pos[0],
                        pos[1],
                        pos[2],
                    ],
                )

        # Create integrator
        integrator = LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
        )

        # Create simulation
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        # Minimize energy
        simulation.minimizeEnergy(maxIterations=200)

        # Get relaxed state
        state = simulation.context.getState(getPositions=True)
        relaxed_positions = state.getPositions()

        # Create a ProDy object for the relaxed structure
        relaxed_structure = original_structure.copy()
        relaxed_structure.setCoords(relaxed_positions.value_in_unit(unit.angstrom))

        # Align relaxed structure to original
        superpose(relaxed_structure, original_structure)

        # Copy B-factors
        relaxed_structure.setBetas(original_structure.getBetas())

        # Write relaxed PDB
        writePDB(relaxed_pdb_path, relaxed_structure)

        clean_pdb(relaxed_pdb_path)


def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Aligns two PDB structures using ProDy.
    """
    ref_struct = parsePDB(reference_pdb)
    align_struct = parsePDB(align_pdb)

    ref_sel_str = f'chain {reference_chain_id.split(",")[0]}'
    align_sel_str = f'chain {align_chain_id.split(",")[0]}'

    ref_sel = ref_struct.select(ref_sel_str)
    align_sel = align_struct.select(align_sel_str)

    # Perform alignment
    transformation = superpose(align_sel, ref_sel)

    # Apply transformation to the whole structure to be aligned
    applyTransformation(transformation, align_struct)

    # Save the aligned structure
    writePDB(align_pdb, align_struct)
    clean_pdb(align_pdb)


def unaligned_rmsd(reference_pdb, align_pdb,
                   reference_chain_id, align_chain_id):
    """
    Calculates the RMSD between two chains without prior alignment.
    """
    ref_struct = parsePDB(reference_pdb)
    align_struct = parsePDB(align_pdb)

    ref_sel = ref_struct.select(f'chain {reference_chain_id} and name CA')
    align_sel = align_struct.select(f'chain {align_chain_id} and name CA')

    rmsd = calcRMSD(align_sel, ref_sel)
    return round(rmsd, 2)


def score_interface(pdb_file, binder_chain="B"):
    """
    Calculates interface scores using a combination of ProDy and BioPython.
    """
    # Load structure with ProDy for interaction analysis
    structure = parsePDB(pdb_file)

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

    except Exception:  # pylint: disable=broad-except-clause
        interface_interface_hbonds = 0
        interface_dg = 0

    # --- SASA Calculation (BioPython) ---
    parser = PDBParser(QUIET=True)
    bp_structure = parser.get_structure("s", pdb_file)
    sr = SASA.ShrakeRupley()
    sr.compute(bp_structure, level="S")

    complex_sasa = bp_structure.sasa

    # Isolate chains for individual SASA calculation
    class ChainSelect:
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            if chain.get_id() == self.chain_id:
                return 1
            return 0

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
    binder_interactions = Interactions(binder_only_structure)
    binder_interactions.calcProteinInteractions(binder_only_structure)
    all_binder_hbonds = binder_interactions.getHydrogenBonds()
    binder_score = 0
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
