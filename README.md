# synrxnforge

**synrxnforge** is a lightweight Python package to generate *synthetic chemical reactions* from a retrosynthetic SMARTS template and a pool of molecules.

It applies reaction templates in reverse (retrosynthetic mode) to product molecules, generating synthetic reaction SMILES datasets — optionally tagged for downstream ML tasks.

Great for creating high volumes of reactions of a given reactivity, should be coupled with some kind of downstream validation process to ensure chemical validity of created reactions. (see *synrxnval*) 

---

## 🚀 Installation

You can install `synrxnforge` directly from source:

```bash
conda create -n synrxnforge python=3.10 -y
conda activate synrxnforge 
conda install -c conda-forge rdkit -y
pip install pytest

git clone https://github.com/yvsgrndjn/synrxnforge.git
cd synrxnforge
pip install -e .
```

