"""
This module provides functions for scoring protein structures and interfaces.
"""
import copy
import warnings
from typing import Tuple, Dict

import numpy as np
from Bio.PDB import NeighborSearch
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.Polypeptide import is_aa

# Try to import freesasa, fall back to Biopython's ShrakeRupley if not available
try:
    import freesasa
    SASA_METHOD = "freesasa"
except ImportError:
    from Bio.PDB.SASA import ShrakeRupley
    SASA_METHOD = "biopython"

from openmm import app, unit, LangevinIntegrator
from openmm.app import PDBFile, ForceField

from pdbfixer import PDBFixer
from .struct_utils import get_chain_atoms, get_chain_residues, load_structure

warnings.filterwarnings("ignore")

# --- Constants ---
HYDROPHOBIC_AA = set('ACFILMPVWY')
AROMATIC_AA = set('FHWY')

# --- Helper Functions ---

def _get_structure_sasa(structure):
    """Calculates SASA for a structure using the best available method."""
    if SASA_METHOD == "freesasa":
        sasa_result, sasa_classes = freesasa.calcBioPDB(structure)
        for atom in structure.get_atoms():
            atom.sasa = sasa_result.atomArea(sasa_classes.atomSerialNumber(atom.serial_number))
    else:
        sr = ShrakeRupley()
        sr.compute(structure, level="A")
    return structure


def calculate_sasa(structure, chain_id=None):
    """Total SASA using the best available method."""
    s = copy.deepcopy(structure)
    _get_structure_sasa(s)
    if chain_id is None:
        return sum(atom.sasa for atom in s.get_atoms())
    total = 0.0
    for model in s:
        if chain_id in [c.id for c in model]:
            for residue in model[chain_id]:
                if is_aa(residue, standard=True):
                    for atom in residue:
                        total += atom.sasa
    return total

def _structure_with_only_chain(structure, keep_chain_id: str):
    s = copy.deepcopy(structure)
    for model in list(s):
        for chain in list(model):
            if chain.id != keep_chain_id:
                model.detach_child(chain.id)
    return s

def calculate_interface_area(structure, chain_a='A', chain_b='B') -> float:
    """ΔSASA/2 = (SASA_A + SASA_B - SASA_AB) / 2"""
    sasa_complex = calculate_sasa(structure)
    mono_a = _structure_with_only_chain(structure, chain_a)
    sasa_a = calculate_sasa(mono_a)
    mono_b = _structure_with_only_chain(structure, chain_b)
    sasa_b = calculate_sasa(mono_b)
    return sasa_a + sasa_b - sasa_complex


def find_interface_residues(structure, chain_a='A', chain_b='B', cutoff=5.0):
    """Use NeighborSearch to find heavy-atom contacts between chains."""
    atoms_a = get_chain_atoms(structure, chain_a, heavy_only=True)
    atoms_b = get_chain_atoms(structure, chain_b, heavy_only=True)
    if not atoms_a or not atoms_b:
        return [], []

    ns = NeighborSearch(list(structure.get_atoms()))
    residues_a = set()
    residues_b = set()
    for atom in atoms_a:
        for neigh in ns.search(atom.coord, cutoff, level='A'):
            if neigh in atoms_b:
                residues_a.add(atom.get_parent())
                residues_b.add(neigh.get_parent())
    return list(residues_a), list(residues_b)


def calculate_hydrogen_bonds(structure, chain_a='A', chain_b='B', distance_cutoff=3.5) -> int:
    """Approximate H-bonds: N/O pairs across interface within cutoff."""
    interface_a, interface_b = find_interface_residues(structure, chain_a, chain_b, cutoff=4.0)
    if not interface_a or not interface_b:
        return 0
    atoms_a = [a for r in interface_a for a in r if a.element in ('N', 'O')]
    atoms_b = [a for r in interface_b for a in r if a.element in ('N', 'O')]
    if not atoms_a or not atoms_b:
        return 0
    ns_b = NeighborSearch(atoms_b)
    hb = 0
    seen = set()
    for a in atoms_a:
        close = ns_b.search(a.coord, distance_cutoff, level='A')
        for b in close:
            key = tuple(sorted((a.serial_number, b.serial_number)))
            if key not in seen:
                seen.add(key)
                hb += 1
    return hb


def calculate_shape_complementarity(structure, chain_a='A', chain_b='B') -> float:
    """Simple complementarity proxy: spread of nearest inter-chain distances at interface."""
    interface_a, interface_b = find_interface_residues(structure, chain_a, chain_b, cutoff=6.0)
    if not interface_a or not interface_b:
        return 0.0
    atoms_b = [a for r in interface_b for a in r if a.element != 'H']
    ns_b = NeighborSearch(atoms_b)
    min_dists = []
    for r in interface_a:
        for a in r:
            if a.element == 'H':
                continue
            neighs = ns_b.search(a.coord, 8.0, level='A')
            if neighs:
                d = min(a - n for n in neighs)
                min_dists.append(d)
    if not min_dists:
        return 0.0
    mean_d = float(np.mean(min_dists))
    std_d = float(np.std(min_dists))
    return max(0.0, 1.0 - (std_d / mean_d)) if mean_d > 0 else 0.0


def calculate_packing_density(structure, chain_a='A', chain_b='B') -> float:
    """Packing proxy: average neighbor count of interface heavy atoms within 4.5 Å."""
    interface_a, interface_b = find_interface_residues(structure, chain_a, chain_b, cutoff=6.0)
    if not interface_a or not interface_b:
        return 0.0
    atoms_all = [a for r in (interface_a + interface_b) for a in r if a.element != 'H']
    ns = NeighborSearch(atoms_all)
    counts = []
    for a in atoms_all:
        neighs = [n for n in ns.search(a.coord, 4.5, level='A') if n is not a]
        counts.append(len(neighs))
    if not counts:
        return 0.0
    return float(np.mean(counts))


def calculate_surface_hydrophobicity(structure, chain_id: str) -> float:
    """Surface hydrophobicity fraction based on residue SASA > threshold."""
    s = copy.deepcopy(structure)
    _get_structure_sasa(s)
    residues = get_chain_residues(s, chain_id)
    if not residues:
        return 0.0
    res_sasa = []
    for r in residues:
        sasa = sum(getattr(a, 'sasa', 0.0) for a in r)
        res_sasa.append((r, sasa))
    surface = [r for (r, sasa) in res_sasa if sasa >= 5.0]
    if not surface:
        return 0.0
    hydrophobic = 0
    for r in surface:
        aa = protein_letters_3to1.get(r.resname, 'X')
        if aa in HYDROPHOBIC_AA or aa in AROMATIC_AA:
            hydrophobic += 1
    return hydrophobic / len(surface) if surface else 0.0


def calculate_openmm_energy(pdb_file: str, chain_id: str) -> float:
    """Calculates the potential energy of a specific chain in a PDB file using OpenMM."""
    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    from io import StringIO
    output = StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, output)
    output.seek(0)
    pdb = PDBFile(output)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    new_topology = app.Topology()
    new_positions = [] * unit.angstroms
    chain = next((c for c in pdb.topology.chains() if c.id == chain_id), None)
    if not chain:
        return 0.0
    new_chain = new_topology.addChain(chain.id)
    atom_map = {}
    for residue in chain.residues():
        new_residue = new_topology.addResidue(residue.name, new_chain, residue.id)
        for atom in residue.atoms():
            new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
            atom_map[atom] = new_atom
            new_positions.append(pdb.positions[atom.index])
    for bond in pdb.topology.bonds():
        if bond[0] in atom_map and bond[1] in atom_map:
            new_topology.addBond(atom_map[bond[0]], atom_map[bond[1]])
    system = forcefield.createSystem(new_topology)
    integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    simulation = app.Simulation(new_topology, system, integrator)
    simulation.context.setPositions(new_positions)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    return energy


def score_interface(pdb_file: str, binder_chain: str = "B", target_chain: str = "A") -> Tuple[Dict[str, float], Dict[str, int], str]:
    """Calculates interface scores using a combination of OpenMM, FreeSASA, and Biopython."""
    structure = load_structure(pdb_file)

    interface_target, interface_binder = find_interface_residues(
        structure, target_chain, binder_chain
    )

    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    interface_residues_pdb_ids = []
    for res in interface_binder:
        aa_1 = protein_letters_3to1.get(res.resname, 'X')
        if aa_1 in interface_AA:
            interface_AA[aa_1] += 1
            res_id = f"{binder_chain}{res.id[1]}"
            interface_residues_pdb_ids.append(res_id)

    interface_nres = len(interface_binder)

    hydrophobic_count = sum(interface_AA[aa] for aa in HYDROPHOBIC_AA if aa in interface_AA)
    interface_hydrophobicity = (hydrophobic_count / interface_nres * 100) if interface_nres > 0 else 0.0

    interface_dSASA = calculate_interface_area(structure, target_chain, binder_chain)
    interface_hbonds = calculate_hydrogen_bonds(structure, target_chain, binder_chain)
    interface_sc = calculate_shape_complementarity(structure, target_chain, binder_chain)
    interface_packstat = calculate_packing_density(structure, target_chain, binder_chain)

    binder_score = calculate_openmm_energy(pdb_file, binder_chain)
    binder_sasa = calculate_sasa(structure, binder_chain)
    interface_binder_fraction = (interface_dSASA / binder_sasa * 100) if binder_sasa > 0 else 0.0

    surface_hydrophobicity = calculate_surface_hydrophobicity(structure, binder_chain)

    # Simple dG proxy
    interface_dG = -(interface_dSASA * 0.01 + interface_hbonds * 2.0 - interface_nres * 0.5)
    interface_dG_SASA_ratio = (interface_dG / interface_dSASA * 100) if interface_dSASA > 0 else 0.0

    interface_hbond_percentage = (interface_hbonds / interface_nres * 100) if interface_nres > 0 else 0.0
    interface_delta_unsat_hbonds = max(0.0, interface_nres * 0.3 - interface_hbonds)
    interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres * 100) if interface_nres > 0 else 0.0

    interface_scores = {
        'binder_score': round(binder_score, 2),
        'surface_hydrophobicity': round(surface_hydrophobicity, 2),
        'interface_sc': round(interface_sc, 2),
        'interface_packstat': round(interface_packstat, 2),
        'interface_dG': round(interface_dG, 2),
        'interface_dSASA': round(interface_dSASA, 2),
        'interface_dG_SASA_ratio': round(interface_dG_SASA_ratio, 2),
        'interface_fraction': round(interface_binder_fraction, 2),
        'interface_hydrophobicity': round(interface_hydrophobicity, 2),
        'interface_nres': interface_nres,
        'interface_interface_hbonds': int(interface_hbonds),
        'interface_hbond_percentage': round(interface_hbond_percentage, 2),
        'interface_delta_unsat_hbonds': round(interface_delta_unsat_hbonds, 2),
        'interface_delta_unsat_hbonds_percentage': round(interface_bunsch_percentage, 2),
    }

    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)
    return interface_scores, interface_AA, interface_residues_pdb_ids_str
