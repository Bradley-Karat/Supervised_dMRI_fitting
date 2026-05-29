import subprocess
import numpy as np
import os

out_dir = config['output_dir']
tmpdir=f"{out_dir}/tmpdir"
modeldir=f"{out_dir}/model"           

if not os.path.isdir(modeldir):           
    subprocess.call(['mkdir', modeldir])

path = expand(inputs['dwi'].input_path,zip,**inputs['dwi'].input_zip_lists)
pathbval = re.sub(".nii.gz", ".bval",path[0])
pathbvec = re.sub(".nii.gz", ".bvec",path[0])

pathcollect = f'{tmpdir}/collect_done.txt'

checkpoint setup_and_train_model:
    input:
        bval = pathbval,       
        bvec = pathbvec,  
        connect = pathcollect     
    params:
        Delta = config["Delta"],
        smalldelta = config["Small_Delta"],
        tmpdir = tmpdir,
        MLmodel = config["ML_model"],
        Nset = config["Nset_size"],
        method = config["MLP_predict_all"],
        Model = config["Model"],
        path_user_signal = config["path_to_noiseless_signal"],
        SM_fit = config["spherical_mean_fit"],
        QCnoise=os.path.join(out_dir,"model","Distribution_of_noise_variances.png"),          
    log:
        log = os.path.join(out_dir,"model","train_machine_learning_model.log"),
    output:
        model=os.path.join(out_dir,"model","trained_model.pkl"),
        modelinfo=os.path.join(out_dir,"model","model_information.pkl"),
        training_set=os.path.join(out_dir,"model","database_training_set.pkl"),
    group:
        "subj"
    script:
        "../scripts/setup_and_run_model_training.py"