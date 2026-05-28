import numpy as np
from normalize_noisemap import normalize_noisemap
from RiceMean import RiceMean
from build_training_set import build_training_set
from train_RF_python import train_RF_python
from train_MLP_python import train_MLP_python
import scipy, scipy.special, scipy.optimize
import random
import time
import pickle
import matplotlib.pyplot as plt
from diffsimgen import diffsimrun
from pathlib import Path
import yaml

# Main script to setup and train the Random
# Forest (RF) or multi-layers perceptron (MLP) regressors used to fit the model

# Author:
# Dr. Bradley Karat
# Department of Radiology
# NYU Grossman School of Medicine
# May 2026
# Email: Bradley.Karat@nyulangone.org

np.random.seed(1)

delta = float(snakemake.params.Delta[0])
smalldel = float(snakemake.params.smalldelta[0])
sigma_mppca = np.load(f'{snakemake.params.tmpdir}/hold_noisemap_norm_mppca.npy')
sigma_SHresiduals = np.load(f'{snakemake.params.tmpdir}/hold_noisemap_norm_SHresiduals.npy')
MLmodel = str(snakemake.params.MLmodel[0])
Nset = float(snakemake.params.Nset)
method = snakemake.params.method
bval = np.round(np.loadtxt(snakemake.input.bval))
bvec = np.loadtxt(snakemake.input.bvec)

bunique = np.unique(bval)

# Sample sigma for Gaussian noise from the sigma distribution from SH residuals
Nsamples = len(sigma_SHresiduals)
sigma_SHresiduals_sampled = sigma_SHresiduals[np.random.randint(0, int(Nsamples), [int(Nset),1], dtype=int)]
sigma_SHresiduals = sigma_SHresiduals_sampled

# If a noisemap from MPPCA denoising is provided, it will use the distribution of noise variances within the volume to add Rician noise floor bias to train the model. If a noisemap is not provided, it will use the user defined SNR.
Nsamples = len(sigma_mppca)
sigma_mppca_sampled = sigma_mppca[np.random.randint(0, int(Nsamples), [int(Nset),1], dtype=int)]
sigma_mppca = sigma_mppca_sampled

modeldict = {"sigma_mppca":sigma_mppca,"sigma_SHresiduals":sigma_SHresiduals,"Nset":int(Nset),"delta":delta,"smalldel":smalldel}

if snakemake.params.Model[0] == 'User_model': # User provided noiseless signal and parameter set. 
    
    noiseless_signal = np.load(snakemake.params.path_user_signal)
    path = Path(snakemake.params.path_user_signal)
    param_file = glob.glob(f"{path.parent}/*parameter*.npy") # Search for corresponding parameter array
    if not param_file:
        raise TypeError("The --Model is User_model but could not find *parameter*.npy file.")
    else:
        parameters = np.load(param_file[0])

    [database_dir_with_rician_bias_noisy, sigma_mppca, Ndirs_per_shell] = build_training_set(bval,bvec,modeldict,noiseless_signal,snakemake.log.log)

    parameter_names = []
    for ii in range(parameters.shape[-1]):
        parameter_names.append(f"parameter_{ii+1}")

    modeldict = {"sigma_mppca":sigma_mppca,"sigma_SHresiduals":sigma_SHresiduals,"Nset":int(Nset),"delta":delta,"smalldel":smalldel,"parameter_names":parameter_names}
    with open(snakemake.output.modelinfo, 'wb') as fp:
        pickle.dump(modeldict, fp)

else:
    if snakemake.params.Model[0] == 'SANDI':
        if np.logical_or(smalldel==0,delta==0):
            raise TypeError("SANDI model was chosen but delta and Delta were not specified. These are required to fit the SANDI model.")
        else:
            signal_directional, signal_powder_avg, noiseless_signal, parameters, parameter_names, SNRarr = diffsimrun(model='SANDI',bval=bval,bvec=bvec,output=None,SNR=[50,100],numofsim=int(Nset),delta=smalldel,Delta=delta)
    else:
        signal, noiseless_signal, parameters, parameter_names, SNRarr = diffsimrun(model=snakemake.params.Model[0],bval=bval,bvec=bvec,output=None,SNR=[50,100],numofsim=int(Nset))
    
    [database_dir_with_rician_bias_noisy, sigma_mppca, Ndirs_per_shell] = build_training_set(bval,bvec,modeldict,noiseless_signal,snakemake.log.log)

if np.logical_or(snakemake.params.SM_fit,snakemake.params.Model[0] == 'SANDI'):

    database_train_noisy = np.zeros((database_dir_with_rician_bias_noisy.shape[0], np.size(bunique)))

    # Identify b-shells and direction-average per shell
    f = open(snakemake.log.log, "a")
    f.write(f"\nDirection-averaging the signals.")
    f.close()

    for i in range(np.size(bunique)):
        database_train_noisy[:,i] = np.nanmean(database_dir_with_rician_bias_noisy[:,bval==bunique[i]],1)

    if 'stick' in snakemake.params.Model[0]:
        parameter_names = parameter_names[2:] # As it stands in diffsimgen, the stick parameters contain theta/phi as first 2 params.
        params_train = parameters[:,2:] 
    else:
        parameter_names = parameter_names
        params_train = parameters

    modeldict = {"sigma_mppca":sigma_mppca,"sigma_SHresiduals":sigma_SHresiduals,"Nset":int(Nset),"delta":delta,"smalldel":smalldel,"parameter_names":parameter_names}
    with open(snakemake.output.modelinfo, 'wb') as fp:
        pickle.dump(modeldict, fp)

    if np.logical_or('Dpar' in parameter_names,'Dperp' in parameter_names):
        ind1 = np.where('Dpar' in parameter_names)
        ind2 = np.where('Dperp' in parameter_names)
        if np.array(ind1).size != 0:
            params_train[:,ind1] = params_train[:,ind1]*1e9
        if np.array(ind2).size != 0:
            params_train[:,ind2] = params_train[:,ind2]*1e9

        params_train[:,ind1] = params_train[:,ind1]*1e9
        params_train[:,ind2] = params_train[:,ind2]*1e9
        
    for i in range(database_train_noisy.shape[1]):
        database_train_noisy[:,i] = database_train_noisy[:,i]/database_train_noisy[:,0]  # Normalize by the b=0

else: # directional signal training, keep orientation params
    
    database_train_noisy = np.zeros((parameters.shape[0], len(bval)))

    b0_mean = np.nanmean(database_dir_with_rician_bias_noisy[:,bval==0],1)
    for i in range(len(bval)):
        database_train_noisy[:,i] = database_dir_with_rician_bias_noisy[:,i]/b0_mean  # Normalize by the b=0

    params_train = parameters

    if 'Dpar' in parameter_names:
        ind1 = parameter_names.index('Dpar')
        params_train[:,ind1] = params_train[:,ind1]*1e9

    if 'Dperp' in parameter_names:
        ind2 = parameter_names.index('Dperp')
        params_train[:,ind2] = params_train[:,ind2]*1e9

    modeldict = {"sigma_mppca":sigma_mppca,"sigma_SHresiduals":sigma_SHresiduals,"Nset":int(Nset),"delta":delta,"smalldel":smalldel,"parameter_names":parameter_names}
    with open(snakemake.output.modelinfo, 'wb') as fp:
        pickle.dump(modeldict, fp)

training_set = {}
training_set['Noisy signal'] = database_train_noisy
training_set['Parameters'] = params_train
with open(snakemake.output.training_set, 'wb') as file:
    pickle.dump(training_set, file)

if MLmodel == 'RF':

    n_trees = 200

    f = open(snakemake.log.log, "a")
    f.write(f'\nTraining a Random Forest with {n_trees} trees for each model parameter...')
    f.close()

    trainedML = train_RF_python((database_train_noisy), params_train, n_trees, snakemake.log.log)
    Mdl = trainedML['Mdl']
    train_perf = trainedML['training_performances']

elif MLmodel == 'MLP':

    n_MLPs = 1 # NOTE: training will take n_MLPs times longer! Training is performed using n_MLPs randomly initiailized for each model parameter. The final prediciton is the average prediciton among the n_MLPS. This should mitigate issues with local minima during training according to the "wisdom of crowd" principle.
    n_layers = 3
    n_units = 30 #3*min(size(database_train_noisy,1),size(database_train_noisy,2)); # Recommend network between 3 x number of b-shells and 5 x number of b-shells

    f = open(snakemake.log.log, "a")
    f.write(f'\nTraining {n_MLPs} MLP(s) with {n_layers} hidden layer(s) and {n_units} units per layer for each model parameter...')
    f.close()
    
    trainedML = train_MLP_python(database_train_noisy, params_train, n_layers, n_units, n_MLPs, method, snakemake.log.log)
    train_perf = trainedML['training_performances']

with open(snakemake.output.model, 'wb') as fp:
    pickle.dump(trainedML, fp)


fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(15,15))

axs = axs.ravel()

axs[0].hist(sigma_mppca_sampled,bins=100)
axs[0].set_title("From the MPPCA noisemap")

axs[1].hist(sigma_SHresiduals_sampled,bins=100)
axs[1].set_title("From the residuals of SH fit")

txt=f"\nDistribution of noise variances. \nThe plot shows the distribution of noise variances used to inject noise to simulated noiseless signals for training. \n[TOP] Noise variance distribution as estimated from MPPCA denoising from the raw data. Median SNR is {(1/np.nanmedian(sigma_mppca_sampled))}. \nThis is used to sample the variance to add the Rician floor bias. \n[BOTTOM] Noise variance distribution as estimated from the SH fit of the fully processed data. Median SNR is {(1/np.nanmedian(sigma_SHresiduals_sampled))}. \nThis is used to add Gaussian noise on top of Rician bias signals, for each acquired direction. "
plt.gcf().text(0.2,0.03, txt)
plt.savefig(snakemake.params.QCnoise)