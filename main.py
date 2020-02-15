from preprocess_data import *
from train_rubbish_detector import *

dataset, num_classes = load_dataset()
train_images, test_images, val_images = split_train_test_val(dataset)


encode_label(list(dataset.keys()), 'metal')
train_data_generator = data_generator("./data/dataset/", list(dataset.keys()), train_images, batch_size)
