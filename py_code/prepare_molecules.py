from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import logging
import sys

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def robust_mol_standardizer(mol, sanitize=True, removeHs=True, strict=False):
    """
    Robust molecule standardizer that handles problematic cases.
    
    Parameters:
    - mol: RDKit molecule
    - sanitize: Attempt to sanitize the molecule
    - removeHs: Remove explicit hydrogens
    - strict: If True, returns None for invalid molecules
    
    Returns:
    - Standardized molecule or None if standardization is not possible
    """
    if mol is None:
        return None
    
    try:
        mol = Chem.Mol(mol)
        
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = fix_valence_problems(mol)
                Chem.SanitizeMol(mol)
        
        if removeHs:
            mol = Chem.RemoveHs(mol)
        
        mol = fix_nitrogen_valence(mol)
        mol = fix_implicit_h_problems(mol)
        mol = standardize_aromaticity(mol)
        
        if sanitize:
            Chem.SanitizeMol(mol)
        
        return mol
    
    except Exception as e:
        if strict:
            logger.error(f"Error standardizing molecule: {str(e)}")
            return None
        return mol

def fix_nitrogen_valence(mol):
    """Fixes nitrogens with invalid valence (4+)"""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetExplicitValence() > 3:
            atom.SetFormalCharge(+1)
            try:
                Chem.SanitizeMol(mol)
            except:
                atom.SetHybridization(Chem.HybridizationType.SP3)
                atom.SetNumExplicitHs(max(0, 3 - atom.GetExplicitValence()))
    return mol

def fix_implicit_h_problems(mol):
    """Removes problematic implicit H groups"""
    mol = Chem.RemoveHs(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    mol = Chem.RemoveHs(mol)
    return mol

def standardize_aromaticity(mol):
    """Standardizes the aromaticity of the molecule"""
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SanitizeMol(mol)
    Chem.GetSymmSSSR(mol)
    mol.UpdatePropertyCache()
    return mol

def fix_valence_problems(mol):
    """Attempts to fix general valence problems"""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetExplicitValence() > 4:
            atom.SetHybridization(Chem.HybridizationType.SP3)
        elif atom.GetAtomicNum() == 8 and atom.GetExplicitValence() > 2:
            atom.SetFormalCharge(-1)
    return mol

def process_sdf_with_retries(input_path, output_path, max_retries=3):
    """
    Processes an SDF file with multiple attempts for problematic molecules.
    """
    supplier = Chem.SDMolSupplier(input_path, sanitize=False)
    writer = Chem.SDWriter(output_path)
    
    success_count = 0
    error_count = 0
    
    for i, mol in enumerate(tqdm(supplier, desc="Processing molecules")):
        if mol is None:
            error_count += 1
            continue
        
        for attempt in range(max_retries):
            try:
                std_mol = robust_mol_standardizer(mol, sanitize=True, removeHs=True, strict=False)
                if std_mol is not None:
                    writer.write(std_mol)
                    success_count += 1
                    break
            except:
                if attempt == max_retries - 1:
                    error_count += 1
                    logger.error(f"Failed to process molecule {i} after {max_retries} attempts")
    
    writer.close()
    logger.info(f"\nProcessing completed: {success_count} molecules standardized, {error_count} failures")
    return success_count, error_count
