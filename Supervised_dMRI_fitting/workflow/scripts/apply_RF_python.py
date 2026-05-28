import numpy as np
import time

def apply_RF_python(signal,trainedML,log):
    
    # Apply pretrained Random Forest regressor

    # Author:
    # Dr. Bradley Karat
    # Department of Radiology
    # NYU Grossman School of Medicine
    # May 2026
    # Email: Bradley.Karat@nyulangone.org

    tic = time.time()

    Mdl = trainedML['Mdl']

    mpgMean = np.zeros((signal.shape[0],len(Mdl)))

    for i in range(len(Mdl)):
        mpgMean[:,i] = Mdl[i].predict(signal)

    toc = time.time()

    tottime = (toc - tic)
    f = open(log, "a")
    f.write(f'\nDONE - RF applied in {round(tottime)} sec.')
    f.close()

    return mpgMean