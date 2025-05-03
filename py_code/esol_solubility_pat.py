import sys
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import namedtuple


class ESOLCalculator:
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def calc_ap(self, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    def calc_esol_descriptors(self, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
        rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_esol(self, mol):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param mol: input molecule
        :return: predicted solubility
        """
        intercept = 0.26121066137801696
        coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
        desc = self.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol


def add_esol_descriptors_to_dataframe(df,smiles_col="SMILES",name_col="Compound ID"):
    """
    Add ESOL descriptors to a Pandas dataframe
    :param df: input dataframe
    :param smiles_col: column in the dataframe with SMILES
    :param name_col: column with the molecule names
    :return: dataframe and list of columns that were added
    """
    esol_calculator = ESOLCalculator()
    PandasTools.AddMoleculeColumnToFrame(df, smiles_col, 'Molecule', includeFingerprints=False)
    result_list = []
    for name, mol in df[[name_col, "Molecule"]].values:
        desc = esol_calculator.calc_esol_descriptors(mol)
        result_list.append([name,desc.mw,desc.logp,desc.rotors,desc.ap])
    result_df = pd.DataFrame(result_list)
    descriptor_cols = ["MW", "Logp", "Rotors", "AP"]
    result_df.columns = [name_col] + descriptor_cols
    df = df.merge(result_df, on=name_col)
    return df, descriptor_cols


def refit_esol(input_file_name, truth_col):
    """
    Refit the parameters for ESOL using multiple linear regression
    Prints parameters that can be pasted into the calc_esol function
    :input_file_name: input file
    :truth_col: column with the experimental value
    :return: None
    """
    df = pd.read_csv(input_file_name)
    df, descriptor_cols = add_esol_descriptors_to_dataframe(df)
    x = df[descriptor_cols]
    y = df[[truth_col]]

    model = LinearRegression()
    model.fit(x, y)
    coefficient_dict = {}
    for name, coef in zip(descriptor_cols, model.coef_[0]):
        coefficient_dict[name.lower()] = coef
    print("intercept = ", model.intercept_)
    print("coef =", coefficient_dict)

