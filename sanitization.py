"""
-------------------------------------------------------------------------------
Script sanitization.py
Purpose RDKit-based sanitization for organometallic complexes.

Key logic:
1. Metal definition, includes transition metals + specific list (Li, Al, Sn, etc.).
2. Dative bonds, auto-converts hypervalent bonds to metals into dative bonds.
3. Charge correction (N, B, P), fixes formal charges if no metal neighbors exist.
4. Platinum correction, strict handling for Pt(II) [4 bonds] and Pt(IV) [6 bonds].

Usage / API
- Import 
    `import sanitization`
- Sanitize string 
    `clean_smi = sanitization.sanitize_smiles(raw_smiles)` 
    -> returns canonical SMILES (str) or None if sanitization fails.
- Sanitize object 
    `clean_mol = sanitization.sanitize_mol_obj(rdkit_mol)` 
    -> returns sanitized RDKit Mol object or None if sanitization fails.
-------------------------------------------------------------------------------
"""

import re
from rdkit import Chem, RDLogger

# Disable unnecessary RDKit logs to avoid console spam during stream processing
RDLogger.DisableLog('rdApp.*')

# --- Constants ---
DATIVE_VALENCE = {
    6: [2, 4],
    7: [3],
    8: [2],
    15: [3, 5],
    16: [2, 4, 6],
    33: [3, 5],
    34: [2, 4, 6]
}

NBP_CHARGES = {5: (4, -1), 7: (4, 1), 15: (6, -1)}

# --- Helper functions ---

def is_transition_metal(at) -> bool:
    n = at.GetAtomicNum()
    return (n >= 21 and n <= 31) or (n >= 39 and n <= 48) or (n >= 71 and n <= 80)

def is_metal(at) -> bool:
    metals = (3, 4, 11, 12, 13, 19, 20, 31, 37, 38, 49, 50, 55, 56, 81, 82, 83, 87, 88)
    n = at.GetAtomicNum()
    return (n in metals) or is_transition_metal(at)

def get_max_valence(at_num: int, actual_val: int) -> int:
    # Original logic without changes
    valence_list = DATIVE_VALENCE[at_num]

    if len(valence_list) == 1:
        return valence_list[0]
    if actual_val in valence_list:
        return actual_val
    if actual_val - 1 in valence_list:
        return actual_val - 1
    if actual_val <= valence_list[0]:
        return valence_list[0]
    if actual_val >= valence_list[-1]:
        return valence_list[-1]

# --- Sanitization steps (with software error protection) ---

def _set_dative_bonds(mol):
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    
    for metal in metals:
        dative_nbr = [at for at in metal.GetNeighbors() if at.GetAtomicNum() in DATIVE_VALENCE]
        for nbr in dative_nbr:
            nbr_atom_num = nbr.GetAtomicNum()
            nbr_actual_valence = nbr.GetExplicitValence()
            nbr_max_valence = get_max_valence(nbr_atom_num, nbr_actual_valence)
            
            bond = rwmol.GetBondBetweenAtoms(nbr.GetIdx(), metal.GetIdx())
            # Software protection check if the bond exists
            if bond is not None:
                bond_type = bond.GetBondType()
                if nbr_actual_valence > nbr_max_valence and bond_type == Chem.BondType.SINGLE:
                    rwmol.RemoveBond(nbr.GetIdx(), metal.GetIdx())
                    rwmol.AddBond(nbr.GetIdx(), metal.GetIdx(), Chem.BondType.DATIVE)
                    
    return rwmol.GetMol() # Returns standard Mol instead of RWMol for stability

def _set_charges_for_nbp(mol):
    mol.UpdatePropertyCache(strict=False)
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in (5, 7, 15)]
    for atom in atoms:
        metals_nbr = [at for at in atom.GetNeighbors() if is_metal(at)]
        num_bonds = atom.GetExplicitValence()
        atom_num = atom.GetAtomicNum()

        if not metals_nbr and num_bonds == NBP_CHARGES[atom_num][0]:
            atom.SetFormalCharge(NBP_CHARGES[atom_num][1])

    return mol

def _set_charges_for_pt(mol):
    # Strict original logic only for 6 and 4 bonds
    mol.UpdatePropertyCache(strict=False)
    pts = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 78]

    for atom in pts:
        num_bonds = len(atom.GetBonds())
        num_non_dative_bonds = len([bond for bond in atom.GetBonds() if bond.GetBondType() != Chem.BondType.DATIVE])

        if num_bonds == 6:
            atom.SetFormalCharge(4 - num_non_dative_bonds)
        elif num_bonds == 4:
            atom.SetFormalCharge(2 - num_non_dative_bonds)

    return mol

# --- Public functions (API) ---

def sanitize_mol_obj(mol):
    """
    Sanitizes an RDKit Mol object.
    Returns a valid Mol or None if sanitization fails.
    """
    if not mol: 
        return None
        
    try:
        # Call order is strictly as in the original script
        mol = _set_dative_bonds(mol)
        mol = _set_charges_for_nbp(mol)
        Chem.SanitizeMol(mol)
        mol = _set_charges_for_pt(mol)
        return mol
    except Exception:
        # If the molecule is unfixable (RDKit crashes) - return None
        return None

def sanitize_smiles(smiles: str) -> str:
    """
    Takes raw SMILES, cleans it, sanitizes the molecule, and returns canonical SMILES.
    Returns None if an error occurs at any stage.
    """
    if not smiles: 
        return None
        
    try:
        # 1. Regex cleaning (from the original script)
        cleaned_smiles = re.sub(r"(?<=[A-Za-z])(-|\+)\d*(?!>)", "", smiles)
        
        # 2. Create object without initial sanitization
        mol = Chem.MolFromSmiles(cleaned_smiles, sanitize=False)
        if not mol: 
            return None
            
        # 3. Apply custom logic
        sanitized_mol = sanitize_mol_obj(mol)
        
        if sanitized_mol:
            return Chem.MolToSmiles(sanitized_mol, isomericSmiles=True)
        else:
            return None
            
    except Exception:
        return None