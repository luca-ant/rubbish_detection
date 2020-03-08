import tensorflow as tf
import os
import re
import traceback
import config
import numpy as np
from preprocess_data import load_labels, load_test_dataset, read_image_as_array
from rubbish_detector_model import restore_model

labels = load_labels(config.labels_file)

def rep_data_gen():
    n = 0
    x_image = list()
    for image_name in load_test_dataset(config.test_dir):
        true_label = re.split(r'[0-9]', image_name)[0]
#        n += 1
#            img = image.load_img(dataset_dir + image_name, target_size=config.input_shape)
#            img = image.img_to_array(img)
        image_array = read_image_as_array(config.test_dir + true_label+ '/'+ image_name)
#        image_array = np.expand_dims(image_array,0)
        x_image.append(image_array)
    images = tf.data.Dataset.from_tensor_slices(np.array(x_image)).batch(1)
    
    for i in images.take(config.batch_size):
        yield [i]
        #if n == config.batch_size:
        #    yield [np.array(x_image)]
        #    x_image = list()
        #    n = 0


opt = {}

opt[''] = {
                    'optimizations':[], 
                    'supported_types':[],
                    'supported_ops': [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS],
                    'representative_dataset': None,
                    'inference_input_type': None,
                    'inference_output_type': None
                    }

opt['_weights-quant'] = {
                    'optimizations':[tf.compat.v1.lite.Optimize.OPTIMIZE_FOR_SIZE], 
                    'supported_types':[],
                    'supported_ops': [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS],
                    'representative_dataset': None,
                    'inference_input_type': None,
                    'inference_output_type': None
                    }

opt['_int-quant'] = {
                    'optimizations':[tf.compat.v1.lite.Optimize.OPTIMIZE_FOR_SIZE], 
#                    'optimizations':[tf.compat.v1.lite.Optimize.DEFAULT], 
                    'supported_types':[],
                    'supported_ops': [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS],
                    'representative_dataset': rep_data_gen,
                    'inference_input_type': None,
                    'inference_output_type': None
                    }

opt['_full-int-quant'] = {
                    'optimizations':[tf.compat.v1.lite.Optimize.OPTIMIZE_FOR_SIZE], 
#                    'optimizations':[tf.compat.v1.lite.Optimize.DEFAULT], 
                    'supported_types':[],
                    'supported_ops': [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS_INT8],
                    'representative_dataset': rep_data_gen,
                    'inference_input_type': tf.uint8,
                    'inference_output_type': tf.uint8
                    }

opt['_float16-quant'] = {'optimizations':[tf.compat.v1.lite.Optimize.DEFAULT], 
                    'supported_types':[tf.float16],
                    'supported_ops': [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS],
                    'representative_dataset': None,
                    'inference_input_type': None,
                    'inference_output_type': None
                    }


if __name__ == "__main__":


#    if os.path.isdir(config.models_dir):
    if os.path.isfile(config.model_file):

#        with os.scandir(config.models_dir) as entries:
#            for e in entries:
#                if e.is_file():
#                    model_file = e.name
#                    model_name = ''.join(model_file.split('.')[0])
                    
                    model_name = config.model_name
                    model_file = model_name+'.h5'

                    for o, params in opt.items():
                        try:
                            model_tflite_file = model_name + o + '.tflite'
#                            converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(config.models_dir+model_file)
                            converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(config.model_file)
                            print("\nCONVERTING {} TO {}\n".format(model_file, model_tflite_file))
                            #Optimization
                            converter.optimizations = params['optimizations']
                            converter.target_spec.supported_types = params['supported_types']
                            converter.target_spec.supported_ops = params['supported_ops']
                            converter.representative_dataset = params['representative_dataset']
                            converter.inference_input_type = params['inference_input_type']
                            converter.inference_output_type = params['inference_output_type']
                            converter.allow_custom_ops = True;
                            tflite_model = converter.convert()

                            os.makedirs(config.models_tflite_dir, exist_ok=True)
                            open(config.models_tflite_dir+model_tflite_file, "wb").write(tflite_model)
                            print('Lite model saved to '+ config.models_tflite_dir+model_tflite_file)
                        except:
                            traceback.print_exc()
                            print('ERROR IN CONVERSION from {} to {}'.format(model_file, model_tflite_file))

    else:
#        print("Models not found in {}".format(config.models_dir))
        print("Model not found in {}".format(config.models_file))
