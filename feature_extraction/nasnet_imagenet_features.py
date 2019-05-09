import tensorflow as tf
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from PIL import Image
from keras import backend as K, backend
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import nasnet
from keras.optimizers import Adadelta
import sys
import numpy as np
import os
from tensorflow.python.client import device_lib


if __name__ == '__main__':
    config=tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction=0.98
    config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU

    sess=tf.Session(config=config)

    filelist = sys.argv[1]
    train_data_dir = sys.argv[2]
    batch_size = int(sys.argv[3])

    inputTensor = Input(shape=(224, 224,3))

    new_model = nasnet.NASNetMobile(input_shape=None, include_top=False, weights='imagenet', input_tensor=inputTensor, pooling=None)
    model = Model(inputs=new_model.input, outputs=new_model.output)

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
            i+=1
        batch = np.asarray(imgbatch)
        print("Processing {}/{} files".format(i,len(files)))

        features = model.predict(batch)
        for k in range(0,len(features)):
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
