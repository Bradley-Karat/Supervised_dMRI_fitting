import subprocess
import numpy as np
import os

out_dir = config['output_dir']
tmpdir=f"{out_dir}/tmpdir"

if not os.path.isdir(tmpdir):           
    subprocess.call(['mkdir', tmpdir])
    np.save(f'{tmpdir}/hold_noisemap_norm_mppca.npy',np.empty((0)))
    np.save(f'{tmpdir}/hold_noisemap_norm_SHresiduals.npy',np.empty((0)))

rule make_noisemap:
    input:
        dwi = inputs.input_path["dwi"],
        mask = re.sub("dwi.nii.gz", "brain_mask.nii.gz",inputs.input_path["dwi"]),       
        bval = re.sub(".nii.gz", ".bval",inputs.input_path["dwi"]),       
        bvec = re.sub(".nii.gz", ".bvec",inputs.input_path["dwi"]),       
        noisemap = re.sub("dwi.nii.gz", "noisemap.nii.gz",inputs.input_path["dwi"]),   
    params:
        FWHM = str(config["FWHM"]),
        SM_fit = config["spherical_mean_fit"],
        direction_avg=bids(
            root="work",
            suffix="diravg_signal.nii.gz",
            datatype="dwi",
            **inputs.input_wildcards["dwi"]
        ),
        tmpdir = tmpdir,
        Model = config["Model"],
    log:
        log = bids(root="logs", suffix="process_all_datasets.log", **inputs.input_wildcards["dwi"]),
    output:
        noisemap=bids(
            root="work",
            suffix="noisemap_from_SHfit.nii.gz",
            datatype="dwi",
            **inputs.input_wildcards["dwi"]
        ),    
        done = touch(temp(bids(
                root="work",
                datatype="dwi",
                suffix="done.txt",
                **inputs.input_wildcards["dwi"]
            ),
        ),
        )
    group:
        "subj"
    script:
        "../scripts/prepare_data_and_noisemaps.py"

pathcollect = f'{tmpdir}/collect_done.txt'

rule collect_done:
    input:
        done = expand(rules.make_noisemap.output.done,zip,**inputs['dwi'].input_zip_lists),
    output:
        collect = pathcollect,
    shell:
        "cat {input.done} > {output.collect}"
