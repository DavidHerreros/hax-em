<h1 align='center'>HAX</h1>

<p align="center">
        
<img alt="Supported Python versions" src="https://img.shields.io/badge/Supported_Python_Versions-3.11-blue">
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

# Reconstruction of motion corrected volumes

- **MoDART**: Real space reconstruction with motion correction derived from deformation fields

# Installation

Hax needs `pip` to install its dependencies. The installation on a independent Conda environment is strongly recommended.

We recommend installing the package directly from Pypi using the command:

```bash
  
  # Cuda 13 command
  pip install hax[cuda13]
  
  # Cuda 12 command
  pip install hax[cuda12]

```

If you prefer to have a local copy of this repository, you may also clone directly from GitHub and install the package with the following command (assuming that you are already inside the cloned folder):
```bash

  pip install .

```

> [!WARNING]
> Supported NVIDIA drivers version: >= 525 (Cuda 12/13 will be installed along the package, so there is no need to have CUDA already installed in your system).


# References

- Herreros, D., Lederman, R.R., Krieger, J.M. et al. **Estimating conformational landscapes from Cryo-EM particles by 3D Zernike polynomials**. *Nat Commun* 14, 154 (2023). 
[![DOI:10.1038/s41467-023-35791-y](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41467-023-35791-y)
- Herreros, D., Kiska, J., Ramirez-Aportela, E. et al. **ZART: A Novel Multiresolution Reconstruction Algorithm with Motion-blur Correction for Single Particle Analysis**. *Journal of Molecular Biology* 435, 168088 (2023). 
[![DOI:10.1016/j.jmb.2023.168088](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1016/j.jmb.2023.168088)
- Herreros, D., Mata, C.P., Noddings, C. et al. **Real-space heterogeneous reconstruction, refinement, and disentanglement of CryoEM conformational states with HetSIREN**. *Nat Commun* 16, 3751 (2025). 
[![DOI:10.1038/s41467-025-59135-0](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41467-025-59135-0)
- Herreros, D., Perez Mata, C., Sanchez Sorzano, C.O. et al. **Merging conformational landscapes in a single consensus space with FlexConsensus algorithm"**. *Nat Methods* (2023). 
[![DOI:10.1038/s41592-025-02841-w](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1038/s41592-025-02841-w)
