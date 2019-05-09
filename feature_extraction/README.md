
# Feature Extraction

## Caffe (GoogLeNet and SqueezeNet)

For experiments with [GoogLeNet](https://arxiv.org/abs/1409.4842) and SqueezeNet, please install [Caffe](http://caffe.berkeleyvision.org/).

For users not familiar with Caffe, we provide a tool for feature extraction: extract_features.py. This tool extract the features directly from the image files, and store the features in separate files for each image contained in the input list. Example of usage: 
```
src/extract_features.py -i extracted_frames/ -l frames.list -o descriptions_dir/ -p caffe/deploy.prototxt -m caffe/model.caffemodel -ms 100 -a caffe/img_mean.binaryproto -ol layer_name -is 224x224 -g -gi 0

-i    Directory with the frames
-l    The list of frames
-o    Output directory name (where you want to save the features extracted)
-p    The deploy.prototxt file path
-m    The architecture weights
-ms   Batch size
-a    Image mean file path (optional)
-ol   Name of the layer you want to use as the feature vector
-is   Input size
-g    Use gpu (default = True)
-gi   GPU id to use
```

**Brief Explanation**: This script is going to instantiate the architecture specified in `deploy.prototxt` in caffe. Then it initializes the network with the weights in `model.caffemodel`, feedforward the images in `frames.list` through the network and, for each input image, save the activations of layer `layer_name` in a file in `descriptions_dir`.

## MobileNetV2 and NASNet (Keras)

For experiments with MobileNetV2 and NASNet, please install [Keras with TensorFlow backend](https://keras.io/).

We developed a python script to run each of our experiments. We explain their usability below:

### ARCHITECTURE_finetuning_features.py:

This script receives a trained keras model (.hdf5), extract features from your data and write a feature file for each input image in the current directory.

This script receives 5 input parameters:

1. File list: Text file with your input data names. One file name in each line, without the full path.
2. Data directory: Path to the directory with the input data.
3. Batch size 
4. Weights file: Path to the trained model (.hdf5)
5. Rewrite your feature files (0 or 1): If your feature files already exist, you can choose whether you want them to be overwritten or not. Sometimes when your script fails in the middle of the feature extraction, choosing not to rewrite the already extracted features can be useful :)

### ARCHITECTURE_imagenet_features.py:

This script loads the imagenet trained model (download automatically), extract features from your input data and write them in the current directory.

This script received 3 input parameters:

1. File list: Text file with your input data names. One file name in each line, without the full path.
2. Data directory: Path to the directory with the input data.
3. Batch size







