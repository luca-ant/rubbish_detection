import os
from dataset import Dataset
from keras.applications import ResNet50
from keras.models import Sequential
from keras.optimizers import Adam
from keras import Input, Model
from keras.layers import Dropout, Dense, Flatten, BatchNormalization
from preprocess_data import *
from keras.engine.saving import load_model, save_model
from keras.callbacks import ModelCheckpoint, Callback


class EpochSaver(Callback):
    def __init__(self, start_epoch, last_epoch_file):
        self.epoch = start_epoch
        self.last_epoch_file = last_epoch_file

    def on_epoch_end(self, epoch, logs={}):
        with open(self.last_epoch_file, "w") as f:
            f.write(str(self.epoch))
        self.epoch += 1



class RubbishDetector():

    __instance = None

    @staticmethod
    def get_instance():

        if RubbishDetector.__instance == None:
            RubbishDetector("./")
        return RubbishDetector.__instance

    def __init__(self, working_dir):
        # private constructor
        if RubbishDetector.__instance != None:
            raise Exception("RubbishDetector class is a singleton! Use RubbishDetector.get_instance()")
        else:
            self.working_dir = working_dir
            self.model_dir = self.working_dir + "model/"
            self.weights_dir = self.working_dir + "weights/"
            self.train_dir= self.working_dir + 'training/'

            self.weights_file = self.weights_dir + "weights.h5"
            self.model_file = self.model_dir + "model.h5"
            self.last_epoch_file = self.train_dir + "last_epoch.txt"
            self.total_epoch_file = self.train_dir + "total_epoch.txt"

            self.dataset = Dataset(self.working_dir)
            self.last_epoch = 0
            self.batch_size = 16
            self.total_epochs = 50
            self.train_images = []
            self.test_images = []
            self.val_images = []

            self.model = None       

            RubbishDetector.__instance = self

    def create_nn(self, num_classes):
        print("CREATING MODEL")
    

        model = Sequential()

        model.add(ResNet50(pooling='avg', weights='imagenet'))  # input_shape = (224,224,3)
        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        print('asdasdasd',num_classes)
        model.add(Dense(num_classes, activation='softmax'))
        
        model.layers[0].trainable = False
        
        opt = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model = model

        self.model.summary()

    def restore_model(self):
        return None

    def train(self):


        opt = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        if self.last_epoch >= self.total_epochs:
            print("LAST EPOCH TOO BIG")
            return

        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
        
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            self.restore_model()

        # callbacks
        save_weights_callback = ModelCheckpoint(self.weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)
        save_epoch_callback = EpochSaver(self.last_epoch + 1, self.last_epoch_file)
        save_model_callback = ModelCheckpoint(self.model_file, verbose=1, period=1)

        # params
        steps_train = (len(self.train_images) // self.batch_size) + 1
        steps_val = (len(self.val_images) // self.batch_size) + 1

        # prepare train and val data generator
        train_data_generator = data_generator(self.dataset.dataset_dir, self.train_images, self.batch_size)
        val_data_generator = data_generator(self.dataset.dataset_dir, self.val_images, self.batch_size)


        print("TRAINING MODEL")
        history = self.model.fit_generator(train_data_generator, epochs=self.total_epochs, steps_per_epoch=steps_train, verbose=2, validation_data=val_data_generator,
                                           validation_steps=steps_val, callbacks=[save_weights_callback, save_model_callback, save_epoch_callback],
                                           initial_epoch=self.last_epoch)

        print("SAVING WEIGHTS TO " + self.weights_file)

        self.model.save_weights(self.weights_file, True)
        print("TRAINING COMPLETE!")

        if os.path.isdir(self.train_dir):
            shutil.rmtree(self.train_dir, ignore_errors=True)

        loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        acc = history.history['acc'][-1]
        val_acc = history.history['val_acc'][-1]
        print(
            "LOSS: {:5.2f}".format(loss) + " - ACC: {:5.2f}%".format(100 * acc) + " - VAL_LOSS: {:5.2f}".format(val_loss) + " - VAL_ACC: {:5.2f}%".format(100 * val_acc))
        return history