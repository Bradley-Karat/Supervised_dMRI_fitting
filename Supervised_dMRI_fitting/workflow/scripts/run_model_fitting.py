import numpy as np
import nibabel as nib
import pickle
from apply_RF_python import apply_RF_python
from apply_MLP_python import apply_MLP_python
import subprocess
import os

# Main script to fit the model using Random
# Forest (RF) and/or multi-layers perceptron (MLP)

# Author:
# Dr. Bradley Karat
# Department of Radiology
# NYU Grossman School of Medicine
# May 2026
# Email: Bradley.Karat@nyulangone.org

np.random.seed(1)

mask_data = snakemake.input.mask
MLmodel = str(snakemake.params.MLmodel[0])
method = snakemake.params.method

Mdl = snakemake.input.model
with open(Mdl, "rb") as fp:
    # Load the model dictionary from the file
    trainedML = pickle.load(fp)

modelinf = snakemake.input.modelinfo
with open(modelinf, "rb") as fp:
    # Load the model information dictionary from the file
    modelinfo = pickle.load(fp)

if np.logical_or(snakemake.params.SM_fit,snakemake.params.Model[0] == 'SANDI'):
    img_data = snakemake.params.direction_avg # data to process: direction-averaged signals for each subject
else:
    img_data = snakemake.params.dwi # data to process: directional diffusion data

bvals = snakemake.input.bvals
bvals = np.loadtxt(bvals)

# Load data

tmpmask = nib.load(mask_data)
mask = np.double(tmpmask.get_fdata())

tmp = nib.load(img_data)
tmpimg = tmp.get_fdata()

I = np.double(tmpimg)
[sx, sy, sz, vol] = I.shape

f = open(snakemake.log.log, "a")
f.write(f"Data {img_data} loaded:")
f.write(f"\nMatrix size = {sx} x {sy} x {sz}")
f.write(f"\nVolumes = {vol}")
f.close()

# Prepare ROI for fitting
ROI = np.reshape(I, [sx*sy*sz,vol])
m = np.reshape(mask, [sx*sy*sz])
signal = (ROI[m==1,:])

if not np.logical_or(snakemake.params.SM_fit,snakemake.params.Model[0] == 'SANDI'): # normalize directional data
    b0mean = np.nanmean(signal[:,bvals<100],axis=1)
    for jj in range(len(bvals)):
        signal[:,jj] = signal[:,jj] / b0mean

# Remove nan or inf and impose that the normalised signal is >= 0
signal[np.isnan(signal)] = 0
signal[np.isinf(signal)] = 0
signal[signal<0] = 0

# Fitting the model to the data using pretrained models
if MLmodel == 'RF':
    f = open(snakemake.log.log, "a")
    f.write(f"\nFitting using a Random Forest regressor")
    f.close()  

    f = open(snakemake.log.log, "a")
    f.write(f"\nApplying the Random Forest...")
    f.close()  

    mpgMean = apply_RF_python(signal, trainedML,snakemake.log.log)

elif MLmodel == 'MLP':

    f = open(snakemake.log.log, "a")
    f.write(f"\nFitting using a MLP regressor")
    f.close()  

    f = open(snakemake.log.log, "a")
    f.write(f"\nApplying the MLP...")
    f.close()  

    mpgMean = apply_MLP_python(signal, trainedML,method,snakemake.log.log)


modelinfo = snakemake.input.modelinfo
outdir = snakemake.output.maps_dir
os.makedirs(outdir, exist_ok=True)

# Load parameter names
with open(modelinfo, "rb") as f:
    info = pickle.load(f)

parameter_names = info["parameter_names"]


# Calculate and save parametric maps
if snakemake.params.Model[0] == 'SANDI':

    mpgMean = np.abs(mpgMean)

    fneu = mpgMean[:,0]
    fsom = mpgMean[:,1]
    fe = 1 - fneu - fsom
    fneurite = fneu / (fneu + fsom + fe)
    fsoma = fsom / (fneu + fsom + fe)
    fextra = fe / (fneu + fsom + fe)

    f = open(snakemake.log.log, "a")
    f.write(f"\nSaving SANDI parameter maps")
    f.close()  

    parameter_names = ['fneurite', 'fsoma', 'Din', 'Rsoma', 'De', 'fextra', 'Rsoma_Low_fsoma_Filtered', 'Din_Low_fsoma_Filtered']

    for i,metric in enumerate(parameter_names):

        itmp = np.zeros((sx*sy*sz))

        if i==0:
            itmp[m==1] = fneurite
        elif i==1:
            itmp[m==1] = fsoma
        elif i==mpgMean.shape[1]:
            itmp[m==1] = fextra
        elif i==mpgMean.shape[1]+1:
            Rsoma_tmp = mpgMean[:,3]
            Rsoma_tmp[fsoma<=0.15] = 0
            itmp[m==1] = Rsoma_tmp
        elif i==mpgMean.shape[1]+2:
            Din_tmp = mpgMean[:,2]
            Din_tmp[fneurite<0.10] = 0
            itmp[m==1] = Din_tmp
        else:
            mpgMean[mpgMean[:,i]<0,i] = 0
            itmp[m==1] = mpgMean[:,i]

        itmp = np.reshape(itmp,[sx,sy,sz])
        imghold = nib.Nifti1Image(itmp,tmpmask.affine)

        outfile = os.path.join(
            outdir,
            f"{metric}.nii.gz"
        )

        nib.save(imghold,outfile)

else:

    mpgMean = np.abs(mpgMean)

    f = open(snakemake.log.log, "a")
    f.write(f"\nSaving {snakemake.params.Model} parameter maps")
    f.close()  

    for i,metric in enumerate(parameter_names):

        itmp = np.zeros((sx*sy*sz))
        itmp[m==1] = mpgMean[:,i]

        itmp = np.reshape(itmp,[sx,sy,sz])
        imghold = nib.Nifti1Image(itmp,tmpmask.affine)

        outfile = os.path.join(
            outdir,
            f"{metric}.nii.gz"
        )

        nib.save(imghold,outfile)

tmpdir = snakemake.params.tmpdir
subprocess.run(['rm','-r', tmpdir])