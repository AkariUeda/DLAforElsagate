import tensorflow as tf
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.models import load_model
from PIL import Image
from keras import backend as K, backend
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import mobilenetv2
from keras.optimizers import Adadelta

import sys
import numpy as np
import os
from tensorflow.python.client import device_lib


if __name__ == '__main__':

    filelist = sys.argv[1]
    train_data_dir = sys.argv[2]
    batch_size = int(sys.argv[3])
    weights = sys.argv[4]
    rewrite = int(sys.argv[5])


    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5  #uncomment to limit GPU allocation

    config.gpu_options.visible_device_list = "0"  #set which GPU to use
    sess=tf.Session(config=config)


    inputTensor = Input(shape=(224, 224,3))


    model = load_model(weights)
    model.summary()
    model.layers.pop()
    model.summary()
    model2 = Model(inputs=model.input, outputs=model.layers[-1].output)

    with open(filelist, 'r') as f:
        files = f.readlines()
    curdir = os.getcwd()

    i=0
    while i<len(files):
        savefiles = []
        imgbatch = []
        completed = 0
        while completed<batch_size and i<len(files):
            filepath = os.path.join(train_data_dir, files[i][:-1])
            imgname = curdir + '/' + files[i][:-1] + '.dsc'
            try:
                if rewrite:
                    image = Image.open(filepath).convert("RGB")
                    image = np.array(image)
                    if(image.shape != (224,224,3)):
                        print("Error: "+imgname)                
                        with open('failed_extractions.txt', 'a') as log:
                            log.write(imgname + '\n')
                    else:
                        savefiles.append(imgname)
                        imgbatch.append(image)
                        completed+=1   
                elif os.path.isfile(imgname) == False:
                    image = Image.open(filepath).convert("RGB")
                    image = np.array(image)
                    if(image.shape != (224,224,3)):
                        print("Error: "+imgname)                
                        with open('failed_extractions.txt', 'a') as log:
                            log.write(imgname + '\n')
                    else:
                        savefiles.append(imgname)
                        imgbatch.append(image)
                        completed+=1   

            except:
                print("Fail: {}".format(imgname))  
                continue             
            i+=1
        batch = np.asarray(imgbatch)
        print("Processing {}/{} files".format(i,len(files)))

        features = model2.predict(batch)
        print(features.shape)
        k=0
        while k < len(features):
            try:
                with open(savefiles[k], 'w') as f:
                    line = '%d 1 1\n'%(len(features[k]))
                    first = True
                    for dsc_value in features[k]:
                        if not first:
                            line += ' '
                        else:
                            first = False
                        line = line + '%.8f'%(dsc_value)
                
                    f.write(line)
                    k += 1
            except:
                print("Failed: {}".format(savefiles[k]))
                continue



