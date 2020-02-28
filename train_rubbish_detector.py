import config
import os
import rubbish_detector_model 
from preprocess_data import data_generator, load_labels, load_train_dataset, load_val_dataset
from keras.engine.saving import load_model, save_model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, CSVLogger, ReduceLROnPlateau



def train(model, labels, train_images, val_images):


    print('\nTrain set size: {}'.format(len(train_images)))
    print('Validation set size: {}\n'.format(len(val_images)))

    if not os.path.isdir(config.train_log_dir):
        os.makedirs(config.train_log_dir)
    if not os.path.isdir(config.models_dir):
        os.makedirs(config.model_dir)

    # callbacks
    save_model_callback = ModelCheckpoint(config.model_checkpoint, monitor='val_accuracy', save_best_only=True, mode='auto',verbose=1, period=1)
#    early_stopping_callback = EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True, verbose=1)
    csv_logger_callback = CSVLogger(config.train_log_file, separator=';', append=False)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_accuracy', mode='auto', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    # params
    steps_train = (len(train_images) // config.batch_size) + 1
    steps_val = (len(val_images) // config.batch_size) + 1

    # prepare train and val data generator
    train_data_generator = data_generator(config.train_dir, labels, train_images, config.batch_size)
    val_data_generator = data_generator(config.val_dir, labels, val_images, config.batch_size)
    

    print("TRAINING MODEL")
    history = model.fit(x=train_data_generator, epochs=config.total_epochs, steps_per_epoch=steps_train, 
                        verbose=1, validation_data=val_data_generator, shuffle=True, validation_steps=steps_val, 
                        callbacks=[save_model_callback, 
                                    # early_stopping_callback, 
                                    reduce_lr_callback,
                                    csv_logger_callback])

#    print("SAVING MODEL TO " + config.model_file)
#    model.save(config.model_file, include_optimizer=False)
    shutil.move(config.model_checkpoint, config.model_file)
    print("TRAINING COMPLETE!")

    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(
        "LOSS: {:5.2f}".format(loss) + " - ACC: {:5.2f}%".format(100 * acc) + " - VAL_LOSS: {:5.2f}".format(val_loss) + " - VAL_ACC: {:5.2f}%".format(100 * val_acc))
    return history




if __name__ == "__main__":

    labels = load_labels(config.labels_file)
    train_images = load_train_dataset(config.train_dir)
    val_images = load_val_dataset(config.val_dir)

    if os.path.isfile(config.model_file):
        model = rubbish_detector_model.restore_model(config.model_file, config.weights_file, len(labels))
    else:
        model = rubbish_detector_model.create_nn(len(labels))


    history = train(model, labels, train_images, val_images)

    print(history)
