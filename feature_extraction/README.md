# Feature Extraction


## Caffe (GoogLeNet and SqueezeNet)

For experiments with [GoogLeNet](https://arxiv.org/abs/1409.4842) and SqueezeNet, please install [Caffe](http://caffe.berkeleyvision.org/).

Caffe framework contains a replication of this architecture, which was employed in this paper. Therefore, for feature extraction, the reader can use Caffe framework with our deploy architecture and one of the CNN pretrained models we provide here.

For users not familiar with Caffe, we provide a tool for feature extraction: extract_features.py. This tool extract the features directly from the image files, and store the features in separate files for each image contained in the input list. Example of usage: 
```
src/extract_features.py -i extracted_frames/ -l frames.list -o descriptions_dir/ -p caffe/deploy.prototxt -m caffe/model.caffemodel -ms 100 -a caffe/img_mean.binaryproto -ol layer_name -is 224x224 -g -gi 0

-i            Directory with the frames
-l            The list of frames
-o            Output directory name (where you want to save the features extracted)
-p            The deploy.prototxt file path
-m            The architecture weights
-ms           Batch size
-a            Image mean file path (optional)
-ol           Name of the layer you want to use as the feature vector
-is           Input size
-g            Use gpu (default = True)
-gi           GPU id to use
```

**Brief Explanation**: This script is going to instantiate the architecture specified in `deploy.prototxt` in caffe. Then it initializes the network with the weights in `model.caffemodel`, feedforward the images in `frames.list` through the network and, for each input image, save the activations of layer `layer_name` in a file in `descriptions_dir`

## Keras (NASNet and MobileNetV2)

For experiments with MobileNetV2 and NASNet, please install [Keras with TensorFlow backend](https://keras.io/).
