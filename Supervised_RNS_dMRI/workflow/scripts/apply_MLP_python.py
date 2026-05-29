import numpy as np
import time

def apply_MLP_python(signal,trainedML,method,log):

    # Apply pretrained Multi Layer Perceptron regressor

    # Author:
    # Dr. Bradley Karat
    # Department of Radiology
    # NYU Grossman School of Medicine
    # May 2026
    # Email: Bradley.Karat@nyulangone.org

    tic = time.time()

    Mdl = trainedML['Mdl']

    if method == 1: #train a single MLP to predict all model parameters

        mpgMean = np.zeros((signal.shape[0],5,len(Mdl)))

        for j in range(len(Mdl)):
            net = Mdl[j]
            mpgMean[:,:,j] = net.predict(signal)

        mpgMean = np.mean(mpgMean,axis=2)
        
    else: #train a single MLP to predict a single model parameter

        nrow = len(Mdl)
        ncol = len(Mdl[0])
        mpgMean = np.zeros((signal.shape[0],nrow,ncol))

        for j in range(ncol):
            for i in range(nrow):
                net = Mdl[i][j]
                mpgMean[:,i,j] = net.predict(signal)
                
        mpgMean = np.mean(mpgMean,axis=2)

    toc = time.time()

    tottime = (toc - tic)
    f = open(log, "a")
    f.write(f'\nDONE - MLP fitted in {round(tottime)} sec.')
    f.close()

    return mpgMean

