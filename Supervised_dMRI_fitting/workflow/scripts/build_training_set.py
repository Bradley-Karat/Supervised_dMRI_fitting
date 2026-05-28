import numpy as np
import time
from RiceMean import RiceMean

def build_training_set(bvals,bvecs,modeldict,noiseless_signal,log):

    # Builds the dataset for supervised training of the machine learning models

    # Author:
    # Dr. Bradley Karat
    # Department of Radiology
    # NYU Grossman School of Medicine
    # May 2026
    # Email: Bradley.Karat@nyulangone.org


    tic = time.time()

    ## Build the training set

    bval_filename = bvals
    bvec_filename = bvecs

    modeldict = modeldict

    sigma_mppca = modeldict['sigma_mppca']
    sigma_SHresiduals = modeldict['sigma_SHresiduals']
    Nset = modeldict['Nset']
    delta = modeldict['delta']
    smalldel = modeldict['smalldel']
    log = log

    # Load bvals and bvecs
    bvals = np.round(bvals/100)*100
    bunique = np.unique(bvals)

    Ndirs_per_shell = np.zeros((len(bunique),1))
    for i in range(len(bunique)):
        Ndirs_per_shell[i] = np.sum(bvals == bunique[i])

    bvecs = bvecs/np.linalg.norm(bvecs) # normalize bvecs
    bvecs[np.isnan(bvecs)] = 0

    database_dir = np.zeros((Nset, len(bvals)))
    database_dir_with_rician_bias = np.zeros((Nset, len(bvals)))
    database_dir_with_rician_bias_noisy = np.zeros((Nset, len(bvals)))

    f = open(log, "a")
    f.write(f"\nAdding Rician bias following sigma distribution from MPPCA across all subjects, with median SNR = {np.nanmedian(1/sigma_mppca)} to the signal for each direction.")
    f.close()

    f = open(log, "a")
    f.write(f"\nAdding Gaussian noise following the distribution from SH residuals, with median SNR = {np.nanmedian(1/sigma_SHresiduals)} to the signal for each direction.")
    f.close()
    
    for i in range(Nset):
        database_dir[i,:] = noiseless_signal[i,:]
        database_dir_with_rician_bias[i,:] = RiceMean(database_dir[i,:], sigma_mppca[i])
        database_dir_with_rician_bias_noisy[i,:] =  database_dir_with_rician_bias[i,:] + sigma_SHresiduals[i]*np.random.normal(0,1,size=len(database_dir_with_rician_bias[i,:]))

    return database_dir_with_rician_bias_noisy, sigma_mppca, Ndirs_per_shell
