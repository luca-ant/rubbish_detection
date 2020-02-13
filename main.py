from dataset import *
from rubbish_detector import RubbishDetector
from preprocess_data import *

rd = RubbishDetector.get_instance()
rd.dataset.load_dataset()
train_images, test_images, val_images = rd.dataset.split_train_test_val()


rd.create_nn()