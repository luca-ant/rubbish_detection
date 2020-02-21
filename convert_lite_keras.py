import tensorflow as tf
import os
import config
from preprocess_data import load_labels
from rubbish_detector_model import convert_model_to_lite, restore_model
labels = load_labels(config.labels_file)


model = restore_model(config.model_file, config.weights_file, len(labels))
convert_model_to_lite(model, config.model_file, config.model_lite_file)
