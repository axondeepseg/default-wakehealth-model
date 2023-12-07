# default-wakehealth-model

## Train and test with nnUNetv2

### Structure of the `scripts` Directory

This directory contains the following components:

- **Conversion Script**: This script, `convert_from_bids_to_nnunetv2_format.py`, is responsible for converting the Wakehealth segmentation dataset from the BIDS format to the format expected by nnUNetv2. The script requires two arguments: the path to the original dataset and the target directory for the new dataset. Here is an example of how to run the script:
```bash
python scripts/convert_from_bids_to_nnunetv2_format.py <PATH/TO/LOCATION/OF/SOURCE/AND/TRAINING/DIRECTORIES/OF/ORIGINAL/DATASET> --TARGETDIR <PATH/TO/NEW/DATASET>
```
For more information about the script and its additional arguments, run the script with the `-h` flag:
```bash
python scripts/convert_from_bids_to_nnunetv2_format.py -h
```
- **Setup Script**: This script sets up the nnUNet environment and runs the preprocessing and dataset integrity verification. To run execute the following command: 
```bash
source scripts/setup_nnunet.sh <PATH/TO/LOCATION/OF/SOURCE/AND/TRAINING/DIRECTORIES/OF/ORIGINAL/DATASET> <PATH/TO/SAVE/RESULTS> [DATASET_ID] [LABEL_TYPE] [DATASET_NAME]
```
- **Training Script**: This script is used to train the nnUNet model. It requires four arguments:
    - `DATASET_ID`: The ID of the dataset to be used for training. This should be an integer.
    - `DATASET_NAME`: The name of the dataset. This will be used to form the full dataset name in the format "DatasetNUM_DATASET_NAME".
    - `DEVICE`: The device to be used for training. This could be a GPU device ID or 'cpu' for CPU, 'mps' for M1/M2 or 'cuda' for any GPU.
    - `FOLDS`: The folds to be used for training. This should be a space-separated list of integers.
To run the training script, execute the following command:
```bash
./scripts/train_nnunet.sh <DATASET_ID> <DATASET_NAME> <DEVICE> <FOLDS...>
```


- **Train Test Split File**: This file is a YAML file that contains the training and testing split for the dataset. It is used by the conversion script above. The file should be named `train_test_split.yaml` and placed in the same directory as the dataset.



### Setting Up Conda Environment

To set up the environment and run the scripts, follow these steps:

1. Create a new conda environment:
```bash
conda create --name wakehealth
```
2. Activate the environment:
```bash
conda activate wakehealth
```
3. Install PyTorch, torchvision, and torchaudio. For NeuroPoly lab members using the GPU servers, use the following command:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
For others, please refer to the PyTorch installation guide at https://pytorch.org/get-started/locally/ to get the appropriate command for your system.

4. Update the environment with the remaining dependencies:
```bash
conda env update --file environment.yaml
```
### Setting Up nnUNet
1. Activate the environment:
```bash
conda activate wakehealth
```

2. To train the model, first, you need to set up nnUNet and preprocess the dataset. This can be done by running the setup script:
```bash
source scripts/setup_nnunet.sh <PATH/TO/LOCATION/OF/SOURCE/AND/TRAINING/DIRECTORIES/OF/ORIGINAL/DATASET> <PATH/TO/SAVE/RESULTS> [DATASET_ID] [DATASET_NAME]
```

### Training nnUNet

After setting up the nnUNet and preprocessing the dataset, you can train the model using the training script. The script requires the following arguments:
- `DATASET_ID`: The ID of the dataset to be used for training. This should be an integer.
- `DATASET_NAME`: The name of the dataset. This will be used to form the full dataset name in the format "DatasetNUM_DATASET_NAME".
- `DEVICE`: The device to be used for training. This could be a GPU device ID or 'cpu' for CPU, 'mps' for M1/M2 or 'cuda' for any GPU.
- `FOLDS`: The folds to be used for training. This should be a space-separated list of integers.
To run the training script, execute the following command:
```bash
./scripts/train_nnunet.sh <DATASET_ID> <DATASET_NAME> <DEVICE> <FOLDS...>
```

## Inference

After training the model, you can perform inference using the following command:
```bash
python scripts/nn_unet_inference.py --path-dataset ${nnUNet_raw}/Dataset<FORMATTED_DATASET_ID>_<DATASET_NAME>/imagesTs --path-out <WHERE/TO/SAVE/RESULTS> --path-model ${nnUNet_results}/Dataset<FORMATTED_DATASET_ID>_<DATASET_NAME>/nnUNetTrainer__nnUNetPlans__2d/ --use-gpu --use-best-checkpoint
```
The `--use-best-checkpoint` flag is optional. If used, the model will use the best checkpoints for inference. If not used, the model will use the latest checkpoints. Based on empirical results, using the `--use-best-checkpoint` flag is recommended.

Note: `<FORMATTED_DATASET_ID>` should be a three-digit number where 1 would become 001 and 23 would become 023.
