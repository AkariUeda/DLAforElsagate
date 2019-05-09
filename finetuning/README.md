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


## SqueezeNet

For experiments with SqueezeNet, please install [Caffe](http://caffe.berkeleyvision.org/).

To train models with caffe, you will need a `solver.prototxt` file that specifies the training parameters and a `train_val.prototxt` file with the architecture setup. Note that in the `solver.prototxt` file, you need to reference the `train_val.prototxt` file. 

In the `squeezenet_solver.prototxt` and `squeezenet_train_val.prototxt` files available in this directory, you can see the files we adapted to our experiments.

The original files were downloaded from the [SqueezeNet's author's github](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1).

You can perform a training executing the following line:
```
caffe train -solver solver.prototxt -weights=weights.caffemodel 2> training.log
```

This line initializes the specified architecture with the weights available in the `.caffemodel` file and, for each epoch, creates a `.caffemodel` file with the model weights and a `.solverstate` file with the training checkpoint.

If you need to resume training from any checkpoint, run the following line:
```
caffe train -solver solver.prototxt -snapshot=file.solverstate 2> resume-training.log
```

