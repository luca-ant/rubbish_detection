import tensorflow as tf
import os
import config
from keras.applications.resnet import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, BatchNormalization, Dropout


def create_nn(num_classes):
    print("CREATING MODEL: "+config.model_name)

    model = Sequential(name=config.model_name)
    
    if config.model_name == 'resnet50':
        model.add(ResNet50(pooling='avg', weights='imagenet'))  # input_shape = (224,224,3)
        opt = Adam(lr=0.0001)
    if config.model_name == 'inceptionV3':
        model.add(InceptionV3(pooling='avg', weights='imagenet'))  # input_shape = (299,299,3)
        opt = Adam(lr=0.0001)
    if config.model_name == 'mobilenetV2':
        model.add(MobileNetV2(pooling='avg', weights='imagenet'))  # input_shape = (224,224,3)
        opt = Adam(lr=0.0001)
    if config.model_name == 'vgg19':
        model.add(VGG19(pooling='avg', weights='imagenet'))  # input_shape = (224,224,3)
        opt = Adam(lr=0.0001)

    model.add(BatchNormalization())
    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(250, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

#    model.layers[0].trainable = False
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
   

    return model

def restore_model(model_file):

    print("LOADING MODEL from "+ model_file)
    model = load_model(model_file)

    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

