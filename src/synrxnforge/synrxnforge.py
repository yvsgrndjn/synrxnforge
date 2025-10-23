from typing import List
import os
import pandas as pd
 
from rdchiral.main import rdchiralRunText
 
from rdkit import Chem
from rdkit.Chem import AllChem
from rxnutils.chem.disconnection_sites.atom_map_tagging import atom_map_tag_reactants, atom_map_tag_products
from rxnutils.chem.disconnection_sites.tag_converting import convert_atom_map_tag
 
# custom function adapted to reactants instead of products
def convert_atom_map_tag_reac_smi(reac_smi: str) -> str:
    """
    Convert atom-map tags for *reactant* SMILES safely, component-wise.

    This adapts `convert_atom_map_tag` (which expects single-component inputs)
    to multi-reactant strings by splitting on '.', converting each component,
    and rejoining.

    Parameters
    ----------
    reac_smi : str
        Reactant SMILES string. Can be multi-component (e.g., "A.B").

    Returns
    -------
    str | None
        Converted SMILES, or `None` if input is NaN/invalid or conversion fails.

    Notes
    -----
    - Failures (e.g., assertion errors from `convert_atom_map_tag`) cause the
      entire multi-component reaction to be marked invalid (returns `None`).
    - The function prints a warning/error in case of failure to aid debugging.
    """
    if pd.isna(reac_smi):
        return None
 
    reac_smi_list = reac_smi.split(".")
    reac_conv = []
 
    for smi in reac_smi_list:
        try:
            conv = convert_atom_map_tag(smi)
        except AssertionError as e:
            print(f"[WARN] Tag conversion failed for SMILES {smi!r}: {e}")
            return None  # mark entire reaction as invalid for filtering later
        except Exception as e:
            print(f"[ERROR] Unexpected error in tag conversion for {smi!r}: {e}")
            return None
 
        reac_conv.append(conv)
 
    return ".".join(reac_conv)



class SynRxnForge:
    """
    apply retrosynthetic templates to molecule pools to generate synthetic reaction SMILES
    """
    def __init__(self, retro_template: str, pool_dataset_path: str = None):
        """
        Parameters
        ----------
        retro_template : str
            Reaction SMARTS template, e.g. '[C:1](=O)O>>[C:1]O'
        pool_dataset_path : str | None
            Optional path to SMILES dataset (one SMILES per line).
            If None, you can directly pass a molecule list to `.main()`.
        """
        self.retro_template = retro_template
        self.pool_dataset_path = pool_dataset_path
 
        self.retro_reac = retro_template.split(">>")[0]
        self.retro_reac_mol = Chem.MolFromSmarts(self.retro_reac)
        self.retro_template_rxn = AllChem.ReactionFromSmarts(self.retro_template)
 
    def load_data_into_df(self, smiles: list[str] | None = None, file_path: str | None = None) -> pd.DataFrame:
        """
        Create a DataFrame containing SMILES.
 
        Parameters
        ----------
        smiles : list[str] | None
            List of SMILES strings to load (takes precedence if given).
        file_path : str | None
            Optional file path to read from if smiles is None.
 
        Returns
        -------
        pd.DataFrame
            A DataFrame with one column: 'smiles'
        """
        if smiles is not None:
            smiles_list = [s for s in smiles if isinstance(s, str) and s.strip()]
        else:
            path = file_path or self.pool_dataset_path
            if not path or not os.path.exists(path):
                raise ValueError("No valid SMILES input provided.")
            with open(path) as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({'smiles': smiles_list})
 
    def convert_smi_to_mol(self, smi: str, sanitize: bool=True):
        """
        Convert a product SMILES string to an RDKit Mol.

        Parameters
        ----------
        smi : str
            SMILES representation of a product molecule.
        sanitize : bool, default=True
            Whether to apply RDKit sanitization (valence checks, aromaticity, etc.).

        Returns
        -------
        rdkit.Chem.rdchem.Mol | None
            Molecule if parsing succeeds, otherwise `None`.

        Notes
        -----
        - Consider turning `sanitize=False` when working with edge-case inputs,
          then sanitizing later in a controlled way.
        """
        try:
            return Chem.MolFromSmiles(smi, sanitize=sanitize)
        except Exception:
            return None
 
    def calc_mol_drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse product SMILES to Mol and drop invalid rows.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'smiles' column of strings.

        Returns
        -------
        pd.DataFrame
            Copy of input with an added 'mol' column (RDKit Mol), filtered so that
            only rows with non-null 'mol' remain. Index is reset.
        """
        df['mol'] = df['smiles'].map(self.convert_smi_to_mol)
        return df[df['mol'].notnull()].reset_index(drop=True)
 
    def get_rows_with_match_substructures(self, df: pd.DataFrame):
        """
        Keep only product molecules that match the LHS SMARTS substructure.

        A product Mol is retained if it contains the substructure defined by
        `self.retro_reac_mol` (compiled from the template LHS).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'mol' column of RDKit molecules.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with only matching molecules.
        """
        patt = self.retro_reac_mol
        mask = df['mol'].apply(lambda mol: mol.HasSubstructMatch(patt))
        return df[mask].reset_index(drop=True)
    
    def assign_atom_maps_to_mol(self, mol: Chem.rdchem.Mol) -> None:
        """
        Assign sequential atom-map numbers (1..N) to all atoms in a product Mol.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Product molecule to modify in place.

        Returns
        -------
        None
        """
        [a.SetAtomMapNum(i+1) for (i, a) in enumerate(mol.GetAtoms())]
 
    def assign_atom_maps_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign atom maps to every product molecule in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'mol' column of RDKit molecules.

        Returns
        -------
        pd.DataFrame
            The same DataFrame (mutated in-place) for chaining.
        """
        for mol in df['mol']:
            if mol is not None:
                self.assign_atom_maps_to_mol(mol)
        return df
 
    def apply_template_on_mol(self, prod_mol: Chem.rdchem.Mol, ) -> List:
        """
        Apply the retrosynthetic template to a single product Mol to generate reactants.

        Parameters
        ----------
        prod_mol : rdkit.Chem.rdchem.Mol
            Product molecule with atom maps assigned.

        Returns
        -------
        List[str]
            List of reactant SMILES (mapped). Empty list if no transformation applies.

        Notes
        -----
        - Uses `rdchiralRunText` with `keep_mapnums=True` to preserve mapping.
        - Any exception results in an empty list and a warning message.
        """
        try:
            map_prod_smi = Chem.MolToSmiles(prod_mol)
            reactants = rdchiralRunText(
                reaction_smarts=self.retro_template,
                reactant_smiles=map_prod_smi,
                keep_mapnums=True,
                )
            if not reactants:
                return []
            return reactants if isinstance(reactants, list) else [reactants]
        except Exception as e:
            print(f"[Warning] Failed applying template {e}")
            return []
   
    def apply_template_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the retrosynthetic template to all product molecules.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a 'mol' column (RDKit Mol).

        Returns
        -------
        pd.DataFrame
            Same DataFrame with an added 'reac' column of `List[str]` (mapped reactants).
        """
        df['reac'] = df['mol'].apply(self.apply_template_on_mol)
        return df
 
    def get_smi_from_mol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert product Mol objects back to SMILES for downstream formatting.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'mol' column.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with an added 'map_prod_smi' column.
        """
        df['map_prod_smi'] = df['mol'].map(Chem.MolToSmiles)
        return df
 
    def format_rxn_smi(self, reac: str, prod: str) -> str:
        """
        Format a mapped reaction SMILES from reactant and product SMILES.

        Parameters
        ----------
        reac : str
            Reactant SMILES (can be multi-component e.g., "A.B").
        prod : str
            Product SMILES.

        Returns
        -------
        str
            Reaction SMILES in the form "reac>>prod".
        """
        return reac + '>>' + prod
   
    def format_rxn_smi_list(self, reac_list: List[str], prod: str) -> List[str]:
        """
        Vectorized formatting for multiple reactant solutions to the same product.

        Parameters
        ----------
        reac_list : List[str]
            List of reactant SMILES candidates.
        prod : str
            Product SMILES (mapped).

        Returns
        -------
        List[str]
            List of reaction SMILES strings (mapped).
        """
        rxn_smi_list = []
        for reac in reac_list:
            rxn_smi_list.append(self.format_rxn_smi(reac, prod))
        return rxn_smi_list
 
    def format_rxn_smi_df(self, df):
        """
        Build mapped reaction SMILES for all rows.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'reac' (List[str]) and 'map_prod_smi' columns.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with a 'map_rxns' column (List[str]).
        """
        rxn_list = []
        for reac_list, prod in zip(df['reac'].values, df['map_prod_smi'].values):
            rxn_list.append(self.format_rxn_smi_list(reac_list, prod))
        df['map_rxns'] = rxn_list
        return df
 
    def tag_rxn_smi(self, rxn_smi: str, alt_tag: bool=True) -> str:
        """
        Generate a *tagged* reaction SMILES from a *mapped* reaction SMILES.

        Tagging can be done in two modes:
        - `alt_tag=True`: convert special tagging to the "!" token notation.
        - `alt_tag=False`: retain atom-map numbering style (e.g., mapnum=1).

        Parameters
        ----------
        rxn_smi : str
            Mapped reaction SMILES "reactants>>product".
        alt_tag : bool, default=True
            If True, convert tags to "!"-notation; otherwise, keep mapnumbers.

        Returns
        -------
        str | None
            Tagged reaction SMILES, or `None` if tagging fails.

        Notes
        -----
        - Uses `rxnutils` functions to derive tags per side, then optionally
          converts to the "!" convention with `convert_atom_map_tag`.
        """
        if not isinstance(rxn_smi, str) or ">>" not in rxn_smi:
            return None
        try:
            map_tag_reac_smi = atom_map_tag_reactants(rxn_smi)
            map_tag_prod_smi = atom_map_tag_products(rxn_smi)
            if alt_tag:
                tag_reac_smi = convert_atom_map_tag_reac_smi(map_tag_reac_smi)
                tag_prod_smi = convert_atom_map_tag(map_tag_prod_smi)
                return tag_reac_smi + ">>" + tag_prod_smi
            else:
                return map_tag_reac_smi + ">>" + map_tag_prod_smi
        except Exception as e:
            print(f"[WARNING] Tagging failed for reaction: {rxn_smi}\n{e}")
            return None
 
    def tag_rxn_smi_list(self, rxn_smi_list: List[str], alt_tag: bool=True) -> List[str]:
        """
        Tag a list of mapped reaction SMILES.

        Parameters
        ----------
        rxn_smi_list : List[str]
            Mapped reaction SMILES strings.
        alt_tag : bool, default=True
            See `tag_rxn_smi`.

        Returns
        -------
        List[str]
            Tagged reaction SMILES strings (None entries may occur if tagging failed).
        """
        tag_rxn_list = []
        for rxn_smi in rxn_smi_list:
            tag_rxn_list.append(
                self.tag_rxn_smi(rxn_smi, alt_tag=alt_tag)
                )
        return tag_rxn_list
 
    def tag_rxn_smi_df(self, df: pd.DataFrame, alt_tag: bool =True) -> pd.DataFrame:
        """
        Tag all mapped reactions in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'map_rxns' column (List[str]).
        alt_tag : bool, default=True
            See `tag_rxn_smi`.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with an added 'tag_rxns' column (List[str]).
        """
        df['tag_rxns'] = df['map_rxns'].apply(self.tag_rxn_smi_list)
        return df
 
    @staticmethod
    def flatten_rxn_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand list-valued reaction columns into a per-reaction flat table.

        Parameters
        ----------
        df : pd.DataFrame
            Expected columns:
            - 'smiles'   : original product SMILES
            - 'map_rxns' : List[str], mapped reaction SMILES
            - 'tag_rxns' : List[str], tagged reaction SMILES (may contain None)

        Returns
        -------
        pd.DataFrame
            Columns:
            - 'smiles' : str, original product SMILES
            - 'map_rxn': str, mapped reaction SMILES
            - 'tag_rxn': str | None, tagged reaction SMILES (may be None)

        Notes
        -----
        - Rows with missing 'map_rxn' are dropped.
        - Index is reset.
        """
        rows = []
        for _, row in df.iterrows():
            map_rxns = row.get('map_rxns', [])
            tag_rxns = row.get('tag_rxns', [])
            for i, map_rxn in enumerate(map_rxns):
                tag_rxn = tag_rxns[i] if i < len(tag_rxns) else None
                rows.append({
                    'smiles': row['smiles'],
                    'map_rxn': map_rxn,
                    'tag_rxn': tag_rxn,
                })
        return pd.DataFrame(rows).dropna(subset=['map_rxn']).reset_index(drop=True)
 
    def save_reactions(self, df: pd.DataFrame, out_dir: str, base_name: str = "synthetic_reactions"):
        """
        Persist flattened reactions to CSV and Parquet.

        Parameters
        ----------
        df : pd.DataFrame
            Output table from `flatten_rxn_columns`.
        out_dir : str
            Directory to write into. Created if it does not exist.
        base_name : str, default="synthetic_reactions"
            Base filename without extension.

        Returns
        -------
        None
        """
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(out_dir, f"{base_name}.parquet")
        #csv_path = os.path.join(out_dir, f"{base_name}.csv")
 
        df.to_parquet(parquet_path, index=False)
        #df.to_csv(csv_path, index=False)
 
        print(f"[INFO] Saved: {parquet_path}")
        #print(f"[INFO] Saved: {csv_path}")
 
    def main(self, out_dir: str = "./outputs", smiles: list[str] | None = None, chunk_id: int | None = None):
        """
        Execute the full SynRxnForge pipeline.

        Parameters
        ----------
        out_dir : str, default="./outputs"
            Output directory for results (CSV, Parquet).
        smiles : list[str] | None
            Optional in-memory list of product SMILES to process. If omitted,
            the instance's `pool_dataset_path` is used.
        chunk_id : int | None
            Optional chunk index. If provided, the output filename is suffixed
            with `_chunk{chunk_id}` for HPC chunking.

        Returns
        -------
        pd.DataFrame | None
            Flattened reactions table (columns: 'smiles', 'map_rxn', 'tag_rxn'),
            or `None` if no matching molecules were found at the screening step.

        Side Effects
        ------------
        Writes `<out_dir>/synthetic_reactions.csv` and `.parquet` to disk.

        Notes
        -----
        - Prints an informational message if screening yields zero matches.
        - This function is intended as the main entry point when using the class.
        """
        df = self.load_data_into_df(smiles=smiles)
        df = self.calc_mol_drop_invalid_rows(df)
        df = self.get_rows_with_match_substructures(df)
        df = self.assign_atom_maps_to_df(df)
 
        if len(df) == 0:
            print("[INFO] No matching molecules found.")
            return None
 
        df = self.apply_template_to_df(df)
        df = self.get_smi_from_mol(df)
        df = self.format_rxn_smi_df(df)
        df = self.tag_rxn_smi_df(df)
 
        flat_df = self.flatten_rxn_columns(df)
        base_name = "synthetic_reactions"
        if chunk_id is not None:
            base_name += f"chunk{chunk_id}"

        self.save_reactions(flat_df, out_dir)
 
        return flat_df
 
if __name__ == "__main__":
    """
    python synrxnforge.py \
        --template "[C:1](=O)O>>[C:1]O" \
        --data ./data/pool.txt \
        --out ./results/    
    """
    import argparse
 
    parser = argparse.ArgumentParser(description="Generate synthetic reactions from a SMARTS template.")
    parser.add_argument("--template", type=str, required=True, help="Reaction SMARTS template.")
    parser.add_argument("--data", type=str, required=True, help="Path to pool dataset (one SMILES per line).")
    parser.add_argument("--out", type=str, default="./outputs", help="Output directory for results.")
    args = parser.parse_args()
 
    synrxnforge = SynRxnForge(
        retro_template=args.template,
        pool_dataset_path=args.data,
    )
 
    final_df = synrxnforge.main(out_dir=args.out)
 
    print(f"[DONE] Generated {len(final_df)} synthetic reactions.")
               
 