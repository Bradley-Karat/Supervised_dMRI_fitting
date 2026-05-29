import os
import subprocess
import pickle

out_dir = config['output_dir']
tmpdir=f"{out_dir}/tmpdir" 

def get_metric_files(wildcards):

    ckpt = checkpoints.setup_and_train_model.get(**wildcards)

    with open(ckpt.output.modelinfo, "rb") as f:
        info = pickle.load(f)

    return expand(
        bids(
            root="results",
            suffix="{metric}.nii.gz",
            desc="ML-fit",
            **inputs.input_wildcards["dwi"]
        ),
        metric=info["parameter_names"]
    )

rule apply_model:
    input:
        model=os.path.join(out_dir,"model","trained_model.pkl"),
        modelinfo=os.path.join(out_dir,"model","model_information.pkl"),
        mask = re.sub("dwi.nii.gz", "brain_mask.nii.gz",inputs.input_path["dwi"]),       
        bvals = re.sub(".nii.gz", ".bval",inputs.input_path["dwi"]),       
    params:
        MLmodel = config["ML_model"],
        method = config["MLP_predict_all"],
        Model = config["Model"],
        direction_avg=bids(
            root="work",
            suffix="diravg_signal.nii.gz",
            datatype="dwi",
            **inputs.input_wildcards["dwi"]
        ),
        dwi = inputs.input_path["dwi"],
        SM_fit = config["spherical_mean_fit"],
        tmpdir = tmpdir
    log:
        log = bids(root="logs", suffix="run_model_fitting.log", **inputs.input_wildcards["dwi"]),
    output:
        maps_dir = directory(
            os.path.join(
                "results",
                "sub-{subject}"
            )
        )
    group:
        "subj"
    script:
        "../scripts/run_model_fitting.py"

rule aggregate_metrics:
    input:
        get_metric_files