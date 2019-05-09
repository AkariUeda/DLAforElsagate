# Finetuning

We performed finetuning for all 4 evaluated architectures. Here we provide a brief explanation on how to run each of our scripts:

## MobileNetV2 and NASNet (Keras)

For experiments with MobileNetV2 and NASNet, please install [Keras with TensorFlow backend](https://keras.io/).

### ARCHITECTURE_imagenet_finetuning.py

This script automatically downloads and initialize the architecture with the ImageNet trained model and finetune it with your input data. For each training epoch, the weights are saved in a .hdf5 file with the name in a format of `weights.EPOCH-VAL_ACC.hdf5`, with the number of the epoch and the validation accuracy in this epoch. These .hdf5 file can be used as a checkpoint and to extract features.

The script has 4 input parameters:

1. Train data directory: Path to the directory with your training set.
2. Validation data directory: Path to the directory with your validation set.
3. Batch size
4. Epochs

### ARCHITECTURE_finetuning_checkpoint.py

1. Train data directory: Path to the directory with your training set.
2. Validation data directory: Path to the directory with your validation set.
3. Batch size
4. Epochs
5. Weights file: The training checkpoint (.hdf5) you want to use to initialize the network to continue the training.



