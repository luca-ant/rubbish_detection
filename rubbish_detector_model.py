import tensorflow as tf
import os
import config
from keras.applications import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Flatten, BatchNormalization, Dropout






def create_nn(num_classes):
    print("CREATING MODEL: "+config.model_name)

    model = Sequential(name=config.model_name)
    
    if config.model_name == 'resnet50':
        model.add(ResNet50(pooling='avg', weights='imagenet'))  # input_shape = (224,224,3)⏎
    if config.model_name == '':
        model.add(InceptionResNetV2(pooling='avg', weights='imagenet'))  # input_shape = (299,299,3)⏎

    model.add(BatchNormalization())
    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(250, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

#    model.layers[0].trainable = False
    
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
   

    return model

def restore_model(model_file):

    print("LOADING MODEL from "+ model_file)
    model = load_model(model_file)

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

