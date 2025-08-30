
"""
Independent Protein Interface Analysis Library (Refactored)
Uses Biopython wherever possible to avoid duplicating calculations.
"""
import copy
import warnings
from typing import List, Tuple, Dict

import numpy as np
from Bio import PDB
from Bio.PDB import NeighborSearch, Selection
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.HSExposure import HSExposureCA

warnings.filterwarnings("ignore")


class ProteinStructureAnalyzer:
    """Main class for protein structure analysis without PyRosetta (Biopython-first)"""

    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)
        self.io = PDB.PDBIO()

        # Amino acid classes
        self.hydrophobic_aa = set('ACFILMPVWY')
        self.polar_aa = set('DEHKNQRST')
        self.aromatic_aa = set('FHWY')
        self.charged_aa = set('DEHKR')

        # Van der Waals radii (Å)
        self.vdw_radii = {
            'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
            'P': 1.80, 'H': 1.09, 'F': 1.47, 'CL': 1.75,
            'BR': 1.85, 'I': 1.98
        }

    # ---------------------------- Core IO ----------------------------
    def load_structure(self, pdb_file: str):
        return self.parser.get_structure('protein', pdb_file)

    # ------------------------- Selection utils -----------------------
    def get_chain_residues(self, structure, chain_id: str):
        residues = []
        for model in structure:
            if chain_id in [c.id for c in model]:
                for residue in model[chain_id]:
                    if is_aa(residue, standard=True):
                        residues.append(residue)
        return residues

    def _get_chain_atoms(self, structure, chain_id: str, heavy_only=True):
        atoms = []
        for model in structure:
            if chain_id in [c.id for c in model]:
                for residue in model[chain_id]:
                    if is_aa(residue, standard=True):
                        for atom in residue:
                            if heavy_only and atom.element == 'H':
                                continue
                            atoms.append(atom)
        return atoms

    def _get_ca_atoms(self, structure, chain_id: str):
        atoms = []
        for model in structure:
            if chain_id in [c.id for c in model]:
                for residue in model[chain_id]:
                    if is_aa(residue) and 'CA' in residue:
                        atoms.append(residue['CA'])
        return atoms

    # --------------------------- SASA --------------------------------
    def _sasa_on_copy(self, structure):
        """Compute SASA with Shrake-Rupley in-place on a deepcopy to avoid side-effects."""
        s = copy.deepcopy(structure)
        sr = ShrakeRupley()
        sr.compute(s, level='A')  # annotate .sasa on atoms
        return s

    def calculate_sasa(self, structure, chain_id: str = None) -> float:
        """Total SASA using Shrake-Rupley."""
        s = self._sasa_on_copy(structure)
        if chain_id is None:
            return sum(getattr(a, 'sasa', 0.0) for a in s.get_atoms())
        total = 0.0
        for model in s:
            if chain_id in [c.id for c in model]:
                for residue in model[chain_id]:
                    if is_aa(residue, standard=True):
                        for atom in residue:
                            total += getattr(atom, 'sasa', 0.0)
        return total

    def _structure_with_only_chain(self, structure, keep_chain_id: str):
        s = copy.deepcopy(structure)
        for model in list(s):
            for chain in list(model):
                if chain.id != keep_chain_id:
                    model.detach_child(chain.id)
        return s

    def calculate_interface_area(self, structure, chain_a='A', chain_b='B') -> float:
        """ΔSASA/2 = (SASA_A + SASA_B - SASA_AB) / 2"""
        complex_s = self._sasa_on_copy(structure)
        sasa_complex = sum(getattr(a, 'sasa', 0.0) for a in complex_s.get_atoms())

        mono_a = self._structure_with_only_chain(structure, chain_a)
        sasa_a = self.calculate_sasa(mono_a)
        mono_b = self._structure_with_only_chain(structure, chain_b)
        sasa_b = self.calculate_sasa(mono_b)
        return (sasa_a + sasa_b - sasa_complex) / 2.0

    # ---------------------- Interface detection ----------------------
    def find_interface_residues(self, structure, chain_a='A', chain_b='B', cutoff=5.0):
        """Use NeighborSearch to find heavy-atom contacts between chains."""
        atoms_a = self._get_chain_atoms(structure, chain_a, heavy_only=True)
        atoms_b = self._get_chain_atoms(structure, chain_b, heavy_only=True)
        if not atoms_a or not atoms_b:
            return [], []

        ns = NeighborSearch(list(structure.get_atoms()))
        residues_a = set()
        residues_b = set()
        for atom in atoms_a:
            # Query neighbors in B within cutoff
            for neigh in ns.search(atom.coord, cutoff, level='A'):
                if neigh in atoms_b:
                    residues_a.add(atom.get_parent())
                    residues_b.add(neigh.get_parent())
        return list(residues_a), list(residues_b)

    # ---------------------- Interface metrics ------------------------
    def calculate_hydrogen_bonds(self, structure, chain_a='A', chain_b='B',
                                 distance_cutoff=3.5) -> int:
        """Approximate H-bonds: N/O pairs across interface within cutoff."""
        interface_a, interface_b = self.find_interface_residues(structure, chain_a, chain_b, cutoff=4.0)
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

    def calculate_shape_complementarity(self, structure, chain_a='A', chain_b='B') -> float:
        """Simple complementarity proxy: spread of nearest inter-chain distances at interface.
        Higher (0-1) is better. Uses NeighborSearch for efficiency.
        """
        interface_a, interface_b = self.find_interface_residues(structure, chain_a, chain_b, cutoff=6.0)
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

    def calculate_packing_density(self, structure, chain_a='A', chain_b='B') -> float:
        """Packing proxy: average neighbor count of interface heavy atoms within 4.5 Å,
        normalized by a heuristic (target ~14 neighbors for dense packing)."""
        interface_a, interface_b = self.find_interface_residues(structure, chain_a, chain_b, cutoff=6.0)
        if not interface_a or not interface_b:
            return 0.0
        atoms_all = [a for r in (interface_a + interface_b) for a in r if a.element != 'H']
        ns = NeighborSearch(atoms_all)
        counts = []
        for a in atoms_all:
            # exclude self
            neighs = [n for n in ns.search(a.coord, 4.5, level='A') if n is not a]
            counts.append(len(neighs))
        if not counts:
            return 0.0
        avg = float(np.mean(counts))
        # 14 is a typical dense packing coordination in proteins (rough heuristic)
        return max(0.0, min(1.0, avg / 14.0))

    def calculate_energy_score(self, structure, chain_id: str) -> float:
        """Coarse intra-chain energy proxy using VDW clashes/contacts with NeighborSearch."""
        atoms = self._get_chain_atoms(structure, chain_id, heavy_only=True)
        if not atoms:
            return 0.0
        ns = NeighborSearch(atoms)
        energy = 0.0
        clash_penalty = 10.0
        contact_bonus = -1.0
        visited = set()
        for a in atoms:
            for b in ns.search(a.coord, 6.0, level='A'):
                if a is b:
                    continue
                key = tuple(sorted((a.serial_number, b.serial_number)))
                if key in visited:
                    continue
                visited.add(key)

                r1 = self.vdw_radii.get(a.element, 1.5)
                r2 = self.vdw_radii.get(b.element, 1.5)
                d = a - b
                vdw = r1 + r2
                if d < 0.8 * vdw:
                    energy += clash_penalty
                elif d < 1.2 * vdw:
                    energy += contact_bonus
        return energy

    def calculate_surface_hydrophobicity(self, structure, chain_id: str) -> float:
        """Surface hydrophobicity fraction based on residue SASA > threshold."""
        s = self._sasa_on_copy(structure)
        # residue SASA
        residues = self.get_chain_residues(s, chain_id)
        if not residues:
            return 0.0
        res_sasa = []
        for r in residues:
            sasa = sum(getattr(a, 'sasa', 0.0) for a in r)
            res_sasa.append((r, sasa))
        # threshold: consider surface if residue SASA >= 5.0 Å^2
        surface = [r for (r, sasa) in res_sasa if sasa >= 5.0]
        if not surface:
            return 0.0
        hydrophobic = 0
        for r in surface:
            try:
                aa = three_to_one(r.resname)
            except KeyError:
                continue
            if aa in self.hydrophobic_aa or aa in self.aromatic_aa:
                hydrophobic += 1
        return hydrophobic / len(surface) if surface else 0.0

    # ------------------------ Scoring pipeline -----------------------
    def score_interface(self, pdb_file, binder_chain='B', target_chain='A'):
        structure = self.load_structure(pdb_file)

        # Interface residues
        interface_target, interface_binder = self.find_interface_residues(
            structure, target_chain, binder_chain
        )

        # Amino acid composition (binder side)
        interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        interface_residues_pdb_ids = []
        for res in interface_binder:
            try:
                aa_1 = three_to_one(res.resname)
            except KeyError:
                aa_1 = 'X'
            if aa_1 in interface_AA:
                interface_AA[aa_1] += 1
                res_id = f"{binder_chain}{res.id[1]}"
                interface_residues_pdb_ids.append(res_id)

        interface_nres = len(interface_binder)

        # Hydrophobicity (%)
        hydrophobic_count = sum(interface_AA[aa] for aa in self.hydrophobic_aa if aa in interface_AA)
        interface_hydrophobicity = (hydrophobic_count / interface_nres * 100) if interface_nres > 0 else 0.0

        # Interface area (ΔSASA)
        interface_dSASA = self.calculate_interface_area(structure, target_chain, binder_chain)

        # Hydrogen bonds
        interface_hbonds = self.calculate_hydrogen_bonds(structure, target_chain, binder_chain)

        # Shape complementarity
        interface_sc = self.calculate_shape_complementarity(structure, target_chain, binder_chain)

        # Packing density proxy
        interface_packstat = self.calculate_packing_density(structure, target_chain, binder_chain)

        # Simplified binding energy (negative favorable)
        interface_dG = -(interface_dSASA * 0.01 + interface_hbonds * 2.0 - interface_nres * 0.5)

        # Ratios
        interface_dG_SASA_ratio = (interface_dG / interface_dSASA * 100) if interface_dSASA > 0 else 0.0

        # Binder metrics
        binder_score = self.calculate_energy_score(structure, binder_chain)
        binder_sasa = self.calculate_sasa(structure, binder_chain)
        interface_binder_fraction = (interface_dSASA / binder_sasa * 100) if binder_sasa > 0 else 0.0

        # Surface hydrophobicity (binder chain)
        surface_hydrophobicity = self.calculate_surface_hydrophobicity(structure, binder_chain)

        # H-bond percentages
        interface_hbond_percentage = (interface_hbonds / interface_nres * 100) if interface_nres > 0 else 0.0

        # Unsatisfied H-bond estimate (simple heuristic)
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

    # ------------------------ Alignment/RMSD -------------------------
    def align_pdbs(self, reference_pdb, align_pdb, reference_chain_id, align_chain_id):
        ref = self.load_structure(reference_pdb)
        mov = self.load_structure(align_pdb)
        ref_atoms = self._get_ca_atoms(ref, reference_chain_id)
        mov_atoms = self._get_ca_atoms(mov, align_chain_id)
        min_len = min(len(ref_atoms), len(mov_atoms))
        ref_atoms = ref_atoms[:min_len]
        mov_atoms = mov_atoms[:min_len]
        sup = Superimposer()
        sup.set_atoms(ref_atoms, mov_atoms)
        sup.apply(mov.get_atoms())
        self.io.set_structure(mov)
        self.io.save(align_pdb)

    def calculate_rmsd(self, reference_pdb, align_pdb, reference_chain_id, align_chain_id) -> float:
        ref = self.load_structure(reference_pdb)
        mov = self.load_structure(align_pdb)
        ref_atoms = self._get_ca_atoms(ref, reference_chain_id)
        mov_atoms = self._get_ca_atoms(mov, align_chain_id)
        min_len = min(len(ref_atoms), len(mov_atoms))
        ref_atoms = ref_atoms[:min_len]
        mov_atoms = mov_atoms[:min_len]
        sup = Superimposer()
        sup.set_atoms(ref_atoms, mov_atoms)
        return round(sup.rms, 2)

    # --------------------------- Relaxation --------------------------
    def relax_structure(self, pdb_file, output_file, iterations=100):
        """Placeholder for relaxation. Copies file to output."""
        import shutil
        shutil.copy(pdb_file, output_file)
        print(f"Structure relaxation placeholder - copied {pdb_file} to {output_file}")


# ------------------------- Wrapper functions -------------------------
def score_interface(pdb_file, binder_chain='B'):
    analyzer = ProteinStructureAnalyzer()
    return analyzer.score_interface(pdb_file, binder_chain)

def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    analyzer = ProteinStructureAnalyzer()
    analyzer.align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id)

def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    analyzer = ProteinStructureAnalyzer()
    return analyzer.calculate_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id)

def pr_relax(pdb_file, relaxed_pdb_path):
    analyzer = ProteinStructureAnalyzer()
    analyzer.relax_structure(pdb_file, relaxed_pdb_path)

def clean_pdb(pdb_file):
    """Clean PDB file (keep ATOM/HETATM/TER/END records)."""
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    cleaned = [ln for ln in lines if ln.startswith(('ATOM', 'HETATM', 'TER', 'END'))]
    with open(pdb_file, 'w') as f:
        f.writelines(cleaned)

if __name__ == "__main__":
    print("Independent Protein Interface Analysis Library (Refactored)")
    print("Successfully loaded - ready to analyze protein structures!")
