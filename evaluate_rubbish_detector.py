from preprocess_data import data_generator, load_labels, load_test_dataset
from keras.preprocessing.image import ImageDataGenerator
import config
import os
import tensorflow as tf
import rubbish_detector_model 




def evaluate(model, labels, test_images):

    print("Test set size: {}".format(len(test_images)))

#    test_data_gen = data_generator(config.test_dir, labels, test_images, config.batch_size)


    image_gen_test = ImageDataGenerator(rescale=1./255)
    test_data_gen = image_gen_test.flow_from_directory(batch_size=config.batch_size,
                                                     directory=config.test_dir,
                                                     target_size=(config.input_shape[0],config.input_shape[1]),
                                                     class_mode='categorical',
                                                     classes=labels
                                                     )

    steps = len(test_images) // config.batch_size + 1
    results = model.evaluate(x=test_data_gen, verbose=1, steps=steps)
    accuracy = results[1]
    print('\nACCURACY: {:.2f}%'.format(accuracy *100))
        


if __name__ == "__main__":


    labels = load_labels(config.labels_file)
    test_images = load_test_dataset(config.test_dir)


    if os.path.isfile(config.model_file):
        model = rubbish_detector_model.restore_model(config.model_file)
        evaluate(model, labels, test_images)
    else:
        print("Model not found in {}".format(config.model_file))
        


