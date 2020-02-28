import tensorflow as tf
import os
import config
from preprocess_data import load_labels
from rubbish_detector_model import restore_model
labels = load_labels(config.labels_file)


#model = restore_model(config.model_file, config.weights_file, len(labels))
#converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model) 


converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(config.model_file)

#Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

os.makedirs(config.model_dir_lite, exist_ok=True)
open(config.model_lite_file, "wb").write(tflite_model)
print('Lite model saved to '+ config.model_lite_file)
