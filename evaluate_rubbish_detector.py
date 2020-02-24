from preprocess_data import data_generator, load_labels, load_test_dataset
import config
import os
import tensorflow as tf
import rubbish_detector_model 




def evaluate(model, labels, test_images):

    print("Test set size: {}".format(len(test_images)))

    test_data_generator = data_generator(config.test_dir, labels, test_images, config.batch_size)
    steps = len(test_images) // config.batch_size + 1
    results = model.evaluate(x=test_data_generator, verbose=1, steps=steps)
    print(results)
        


if __name__ == "__main__":


    labels = load_labels(config.labels_file)
    test_images = load_test_dataset(config.test_dir)


    if os.path.isdir(config.model_dir):
        model = rubbish_detector_model.restore_model(config.model_file, config.weights_file, len(labels))
        evaluate(model, labels, test_images)
    else:
        print("Model not found in {}".format(config.model_dir))
        


