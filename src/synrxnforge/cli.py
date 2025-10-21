"""
Command-line interface for the SynRxnForge package.

This module allows users to generate synthetic reaction SMILES datasets
from retrosynthetic SMARTS templates and pools of molecules directly
from the command line.

Example:
--------
$ synrxnforge --template "[C:1](=O)O>>[C:1]O" \
              --input ./data/pool.txt \
              --output ./results
"""

import argparse
from .synrxnforge import SynRxnForge


def main():
    """
    Entry point for the SynRxnForge command-line interface.
    
    Parses user-provided arguments and executes the main generation pipeline.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic reactions from a SMARTS template.")
    parser.add_argument("--template", type=str, required=True, help="Retrosynthetic reaction SMARTS template.")
    parser.add_argument("--input", type=str, required=True, help="Path to pool dataset (one SMILES per line).")
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results.")
    args = parser.parse_args()

    synrxnforge = SynRxnForge(args.template, pool_dataset_path=args.input)
    synrxnforge.main(out_dir=args.output)


if __name__ == "__main__":
    main()
