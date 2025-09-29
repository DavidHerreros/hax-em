<h1 align='center'>HAX</h1>

<p align="center">
        
<img alt="Supported Python versions" src="https://img.shields.io/badge/Supported_Python_Versions-3.8_%7C_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue">
<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/I2PC/Flexutils-Toolkit/total">
<img alt="GitHub License" src="https://img.shields.io/github/license/I2PC/Flexutils-Toolkit">

</p>

<p align="center">
        
<img alt="HAX" width="300" src="hax/viewers/annotate_space/media/logo.png">

</p>

This package includes several tools to study conformational heterogeneity from CryoEM data.

# Heterogeneity analysis programs

- **Zernike3Deep**: Semi-classical neural network to analyze continuous heterogeneity with the Zernike3D basis
- **HetSIREN**: Neural network heterogeneous reconstruction for real space
- **ReconSIREN**: Neural network for ab initio reconstruction and global angular assignment

# Consensus of conformational landscapes

- **FlexConsensus**: Consensus neural network for conformational landscapes

# Annotation of conformational landscapes

- **Annotate space**: Interactive inspection of conformational landscapes and conformational states

# Installation

Hax needs `pip` to install its dependencies. The installation on a independent Conda environment is strongly recommended.

We recommend installing the package directly from Pypi using the command:

```bash

  pip install hax

```

If you prefer to have a local copy of this repository, you may also clone directly from GitHub and install the package with the following command (assuming that you are already inside the cloned folder):
```bash

  pip install .

```


