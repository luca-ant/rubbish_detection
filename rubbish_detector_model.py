import tensorflow as tf
import os
from keras.applications import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, BatchNormalization
import config

def create_nn(num_classes):
    print("CREATING MODEL")

    model = Sequential()

#    model.add(ResNet50(include_top=True, weights=None, classes=num_classes))  # input_shape = (224,224,3)⏎
#    model.add(InceptionResNetV2(include_top=True, weights=None, classes=num_classes))  # input_shape = (299,299,3)⏎

#    model.add(ResNet50(pooling='avg', weights='imagenet'))  # input_shape = (224,224,3)⏎
    model.add(InceptionResNetV2(pooling='avg', weights='imagenet'))  # input_shape = (299,299,3)⏎
    model.add(Dense(500, activation='relu'))
#    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(250, activation='relu'))
#    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(num_classes, activation='softmax'))

    model.layers[0].trainable = False
    
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
   

    return model

def restore_model(model_file, weights_file, num_classes):

    #model = create_nn(num_classes)
    #print("LOADING WEIGHTS")
    #model.load_weights(weights_file)
    #opt = Adam(lr=0.0001)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("LOADING MODEL")
    model = tf.keras.models.load_model(model_file)
    model.layers[0].trainable = False
    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def convert_model_to_lite(model, model_file, model_lite_file):

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file)
#    converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model) 
    tflite_model = converter.convert()

    os.makedirs(config.model_dir_lite, exist_ok=True)
    open(model_lite_file, "wb").write(tflite_model)
    print('Lite model saved to '+ model_lite_file)
