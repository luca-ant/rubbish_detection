from preprocess_data import split_train_test_val, data_generator, load_dataset
import config
import os
import tensorflow as tf
import rubbish_detector_model 




def evaluate(model, labels, test_images):

    print("Test set size: {}".format(len(test_images)))

    test_data_generator = data_generator(config.dataset_dir, labels, train_images, config.batch_size)
    steps = len(test_images) // config.batch_size + 1
    results = model.evaluate(x=test_data_generator, verbose=1, steps=steps)
    print(results)
        


if __name__ == "__main__":


    dataset, labels = load_dataset(working_dir=config.working_dir)
    train_images, test_images, val_images = split_train_test_val(dataset)


    if os.path.isdir(config.model_dir):
        model = rubbish_detector_model.restore_model(config.model_file, config.weights_file, len(labels))


    evaluate(model, labels, test_images)
