# Feature Extraction

## Software Requirements

* **Caffe**: For experiments with [GoogLeNet](https://arxiv.org/abs/1409.4842) and SqueezeNet, please install caffe.
* **Keras**: For experiments with MobileNetV2 and NASNet, please install Keras with TensorFlow backend.

## GoogLeNet

* The imagenet weights are available at the [bvlc googlenet repository](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel).

* The weights finetuned for pornography in natural videos can be found in `caffe/googlenet_imagenet_porn`

* The weights finetuned for pornography in natural videos can be found in `caffe/googlenet_imagenet_elsagate`

Feature extraction, of the data obtained from the video, is performed using the GoogleNet CNN architecture. Caffe framework contains a replication of this architecture, which was employed in this paper. Therefore, for feature extraction, the reader can use Caffe framework with our deploy architecture (For early fusion color, use this architecture instead) and one of the CNN pretrained models we provide here. The features used in the paper came from the layer named "pool5/7x7_s1".

For users not familiar with Caffe, we provide a tool for feature extraction: extract_features.py. This tool extract the features directly from the image files, and store the features in separate files for each image contained in the input list. Example of usage: 
```
src/extract_features.py -i extracted_frames/ -l frames.list -o dsc/ -p caffe/deploy.prototxt -m caffe/models/raw_frames/s1_x1.caffemodel -ms 100 -a caffe/imgs_mean/raw_frames/s1_x1.binaryproto -ol pool5/7x7_s1 -is 224x224 -g -gi 0
```

## SqueezeNet



## MobileNetV2


## NASNet