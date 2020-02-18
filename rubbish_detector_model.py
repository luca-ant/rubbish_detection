import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, BatchNormalization


def create_nn(num_classes):
    print("CREATING MODEL")

    model = Sequential()

    model.add(ResNet50(pooling='avg', weights='imagenet'))  # input_shape = (224,224,3)
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.layers[0].trainable = False
    
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
   

    return model

def restore_model(model_file, weights_file, num_classes):

    model = create_nn(num_classes)
    print("LOADING WEIGHTS")
    model.load_weights(weights_file)
    #opt = Adam(lr=0.001)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    #model = tf.keras.models.load_model(model_file, compile=True)
    #model.layers[0].trainable = False
    #model.summary()

    return model