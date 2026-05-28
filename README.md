# Supervised ML dMRI toolbox
A BIDS app for fitting diffusion MRI models using supervised machine learning with a realistic noise synthesis approach (see reference below). The training signal can be given by the user (see the -h flag and the --Model flag) or a predefined model can be selected.

## Installation
Dependency management and python package handling is done with [poetry](https://python-poetry.org/docs/).
```
pip install poetry
```
then the toolbox can be installed with:
```
git clone https://github.com/Bradley-Karat/Supervised_dMRI_fitting.git
cd Supervised_dMRI_fitting
poetry install
```
## Usage
The package can be ran with:
```
poetry run Supervised_dMRI_fitting
```
or 
```
poetry shell
Supervised_dMRI_fitting
```
Run either of these with the -h flag to get a detailed summary of the toolbox and its flags plus required arguments.
## Example Toolbox Call
For a dry-run:
```
Supervised_dMRI_fitting /path/to/bids/inputs /path/for/outputs participant --Model SANDI --Delta 23.6 --Small_Delta 7 --cores all -np
```
To actually run the software:
```
Supervised_dMRI_fitting /path/to/bids/inputs /path/for/outputs participant --Model SANDI --Delta 23.6 --Small_Delta 7 --cores all
```

## Example BIDS File Struture
```
└── bids
    ├── sub-01
    │ └── dwi
    │     ├── sub-01_brain_mask.nii.gz
    │     ├── sub-01_dwi.bval
    │     ├── sub-01_dwi.bvec
    │     ├── sub-01_dwi.nii.gz
    │     └── sub-01_noisemap.nii.gz
    ├── sub-02
    │ └── dwi
    │     ├── sub-02_brain_mask.nii.gz
    │     ├── sub-02_dwi.bval
    │     ├── sub-02_dwi.bvec
    │     ├── sub-02_dwi.nii.gz
    │     └── sub-02_noisemap.nii.gz
```
Alternatively, the `--path-dwi` flag can be used to override BIDS by specifying absolute paths. For example: `--path-dwi /path/to/my_data/{subject}/dwi.nii.gz`. The other necessary files (.bval, .bvec, noisemap and brainmask) must be labelled in a similar fashion (i.e. all have the same prefix). In this example of `--path-dwi`, the files would be named dwi.bval, dwi.bvec, noisemap.nii.gz, and brain_mask.nii.gz.

## Reference
If you use this toolbox or its approach for noise addition, please cite:
(preprint)
