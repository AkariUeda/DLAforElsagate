# Classification

## Software Requirements

We used [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) version 3.18.

However, we used a linear kernel SVM, not present in the download link above. In the libsvm3.18 folder of this repository you will find the script `easy_linear.py` in addition to all original libsvm scripts without modifications.

To run the classification, you will need a train_set_features.txt and a test_set_features.txt file. Is up to you how you should split the sets and to generate these files. 


Also, note that these files need to be in the libsvm format. Using a single line for each input image, write an integer indicating the ground truth and the features. In the example below, 1 indicates "elsagate" and -1 "safe" and each of the following data indicates the feature `i` and the value of feature `i`.
```
1 1:0.00008904 2:0.00289221 3:0.06509972 4:0.00028156 5:0.40549973 ... 1000:0.1039836 
-1 1:0.00008904 2:0.00289221 3:0.06509972 4:0.00028156 5:0.40549973 ... 1000:0.1039836 
```

**Important**: Always write the positive class lines before the negative class'. For instance, if you have 40 positive class videos and 20 negative class videos, write the 40 lines belonging the positive videos first.

If you use the script `pooler.py` explained in the [feature extraction section](https://github.com/AkariUeda/DLAforElsagate/tree/master/feature_extraction), the features will be generated in this format. But you still need to split them into training and test sets and make the positive class videos appear before the negative videos.

### Training:
Execute the script below, giving as an input parameter the training set features. It is going to generate some auxiliary files in the current directory that will be used in the next steps, such as the trained model and the range values to properly scale the data.

```
libsvm-3.18/tools/easy_linear.py videos_train.txt
```

### Testing
Using the `.range` files generated in the training step above, we need to scale the test data so that they will be in the same scale as the training data:
```
libsvm-3.18/svm-scale -r videos_train.range videos_test.txt > videos_test.scale
````

In the file `videos_test.scale` you will now have your test set features scaled.

Now use the trained model to predict your data:
```
libsvm-3.18/svm-predict -b 1 videos_test.scale videos_train.model videos_test.predict
```

In the `videos_test.predict` file, you will have for each video a line as the following example:
```
1 0.5 0.6
```

Where the first number is the predicted class (i.e., 1 or -1), the score for the first class and the score for the second class. The first number (i.e., the predicted class) is defined by the maximum score between the classes.

### Fusion

If you are using frames and motion vectors, you are supposed to have now two `.predict` files, with the classification score regarding the frames and the motion vectors.

You can combine these scores to form a fused classification running the following script:

```
java -jar LateFusionp7.jar -f1 video_frames.predict -f2 video_motions.predict -o fusion.predict -f avg
```

Now the `fusion.predict` will have the fusion scores. 

### Analyzing the data

You can run the following script to get the confusion matrix values.

Use as input the `.predict` file generated above and the **number of the first video of the negative class**. For instance, if you have 40 positive class videos and 20 negatives, you need to run the following line using X=41.

```
libsvm_prediction_analyser/build/libsvm_prediction_analyser -i videos.predict -n X
```











