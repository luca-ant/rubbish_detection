import tensorflow as tf
import os
import config

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(config.model_dir+'model.h5')
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

os.makedirs(config.working_dir+'model_lite')
open(config.working_dir+"model_lite/model_lite.tflite", "wb").write(tflite_model)
