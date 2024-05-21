# DS-3D-UNet
Detail-Sensitive 3D-UNet for Pulmonary Airway Segmentation from CT images

Contact:Jiajie Li(15316072595@163.com)

## Model Architecture

![image](https://github.com/nighlevil/DS-3D-UNet/tree/master/DS-3D-UNet/model_architecture.png)

# Requirements
- python packages required are in "requirements.txt"
- cuda >= 10.2 (https://developer.nvidia.com/cuda-zone)
- cuDNN >= 7.6.5 (https://developer.nvidia.com/rdp/cudnn-download)

(Recommended to use python virtualenv)

- python -m venv <path_new_pyvenv>
- source <path_new_pyvenv>/bin/activate
- pip install -r requirements.txt

# Instructions
------------

## Prepare Data Directory

Before running the scripts, the user needs to prepare the data directory with the following structure:

    ├── Images                  <- Store CT scans (in dicom or nifti format)
    ├── Airways                 <- Store reference airway segmentations
    ├── Lungs (optional)        <- Store lung segmentations 
    │                              (used in options i) mask ground-truth to ROI, and ii) crop images)

## Prepare Working Directory

The user needs to prepare the working directory in the desired location, as follows:

1. mkdir <path_your_work_dir> && cd <path_your_work_dir>
2. ln -s <path_your_data_dir> BaseData
3. ln -s <path_this_repo> Code

## Run the Scripts

The user needs only to run the scripts in the directories: "scripts_evalresults", "scripts_experiments", "scripts_launch", "scripts_preparedata", "scripts_util". Each script performs a separate and well-defined operation, 
either to i) prepare data, ii) run experiments, or iii) evaluate results.

(IMPORTANT): set the variable PYTHONPATH with the path of this code as follows:

- export PYTHONPATH=<path_this_repo>/src/

## Important Scripts 

### Steps to Prepare Data

1\. From the data directory above, create the working data used for training / testing:

- python <path_this_repo>/src/scripts_preparedata/prepare_data.py --datadir=<path_data_dir>

Several preprocessing operations can be applied in this script:

1. mask ground-truth to ROI: lungs
2. crop images around the lungs
3. rescale images

IF use option to crop images: compute the bounding-boxes of the lung masks, prior to the script above:

- python <path_this_repo>/src/scripts_preparedata/compute_boundingbox_images.py --datadir=<path_data_dir> 

### Steps to Train Models

1\. Distribute the working data in training / validation / testing:

- python <path_this_repo>/src/scripts_experiments/distribute_data.py --basedir=<path_work_dir>

2\. Launch a training experiment:

- python <path_this_repo>/src/scripts_experiments/train_model.py --basedir=<path_work_dir> --modelsdir=<path_output_models> 

OR restart a previous training experiment:

- python <path_this_repo>/src/scripts_experiments/train_model.py --basedir=<path_work_dir> --modelsdir=<path_stored_models> --is_restart=True --in_config_file=<path_config_file>

### Steps to Test Models

Compute probability maps from a trained model:

- python <path_this_repo>/src/scripts_experiments/predict_model.py <path_trained_model> <path_output_work_probmaps> --basedir=<path_work_dir> --in_config_file=<path_config_file>)

The output prob. maps have the format and dimensions as the working data used for testing, which is typically different from that of the original data (if using options above for preprocessing in the script "prepare_data.py").

#   Citing DS-3D-UNet

not yet pubilshed

# License

The code is released under the [MIT license](./DS-3D-UNet/LICENSE)

# Source Code

Source code:
BronchiNet: https://github.com/antonioguj/bronchinet
