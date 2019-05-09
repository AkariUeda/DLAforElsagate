import tensorflow as tf
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from PIL import Image
from keras import backend as K, backend
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import nasnet
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from time import time

import sys
import numpy as np
import os
from tensorflow.python.client import device_lib


if __name__ == '__main__':


    train_data_dir = sys.argv[1]
    validation_data_dir = sys.argv[2]
    batch_size = int(sys.argv[3])
    epochs = int(sys.argv[4])

    # source activate Tensorflow

    ### start session
    config=tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction=0.98
    config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU

    sess=tf.Session(config=config)

    img_width, img_height = 224, 224
    inputTensor = Input(shape=(224, 224,3))

    new_model = nasnet.NASNetMobile(input_shape=None, include_top=False, weights='imagenet', input_tensor=inputTensor, pooling='avg')
    new_model.summary()
    x = new_model.output

    #x = Dense(224, activation='relu', name='fc2')(output_model)
    x = Dense(2, activation='softmax', name='predictions')(x)

    model = Model(inputs=new_model.input, outputs=x)
    adadelta = Adadelta(lr=0.5, rho=0.95, epsilon=1e-6)
    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer=adadelta,
                  metrics=['mean_squared_error','accuracy'])

    #Callbacks

    filepath = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'
    saveModels = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger('training.log')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    train_datagen = ImageDataGenerator()

    # this is the augmentation configuration we will use for testing:
    # only rescaling

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)


    model.fit_generator(
    train_generator,
    steps_per_epoch=batch_size//2,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[saveModels, tensorboard, csv_logger],
    validation_steps=batch_size//4 )

    model.save_weights('first_try1.h5')
