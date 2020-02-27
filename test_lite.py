import tensorflow as tf
import numpy as np
import cv2
import config
import re
import os
import time
from progress.bar import Bar
from predict_rubbish_lite import predict_class_lite
from preprocess_data import decode_label, load_labels, load_test_dataset, read_image_as_array


height = config.input_shape[0]
width = config.input_shape[1]






def test(name, interpreter, test_images, labesl):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accurate_count = 0
    total_time = 0

    bar = Bar('Testing model lite '+ name, max=len(test_images))

    for image_name in test_images:
        true_label = re.split(r'[0-9]', image_name)[0]
        image_array = read_image_as_array(config.test_dir+image_name)

        img_batch = np.expand_dims(image_array, 0)


        interpreter.set_tensor(input_details[0]['index'], img_batch)
        
        start = time.time()
        interpreter.invoke()
        stop = time.time()
        total_time += stop - start

        output_data = interpreter.get_tensor(output_details[0]['index'])

#       predictions = dict(zip(labels, list(output_data[0])))
#       predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
#       print(' '+'='*21+' ')
#       for k, v in predictions.items():
#           print('|{:12}| {:6.2f}%|'.format(k, v*100))
#        print(' '+'='*21)
        
        if true_label.strip() == decode_label(labels, output_data).strip():
            accurate_count += 1
        bar.next()
    bar.finish()

    accuracy = accurate_count * 1.0 / len(test_images)
    print('\nACCURACY: {:.2f}%'.format(accuracy *100))
    print('TIME/IMAGE: {:.5f} us'.format(total_time /len(test_images)))

    os.makedirs(config.test_res_dir_lite, exist_ok=True)
    with open(config.test_res_dir_lite+name+'_results.txt', "w") as f:
        f.write('ACCURACY: {:.2f}%\n'.format(accuracy *100))
        f.write('TIME/IMAGE: {:.5f} us'.format(total_time /len(test_images)))



if __name__ == "__main__":

    labels = load_labels(config.labels_file)
    test_images = load_test_dataset(config.test_dir)

    optimizations = {}

    optimizations['default_opt'] =[tf.lite.Optimize.DEFAULT]
    optimizations['size_opt'] =[tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    optimizations['lat_opt'] =[tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    optimizations['all_opt'] =[tf.lite.Optimize.OPTIMIZE_FOR_SIZE, tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]



    if os.path.isfile(config.model_file):

        for name, opts in optimizations.items():

            current_model_lite_file = 'rd_model_lite_'+name +'.tflite'

            converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(config.model_file)
             converter.experimental_new_converter = True
            converter.optimizations = opts
            tflite_model = converter.convert()
            
            os.makedirs(config.model_dir_lite, exist_ok=True)
            open(config.model_dir_lite+current_model_lite_file, "wb").write(tflite_model)
            print('Lite model saved to '+config.model_dir_lite+current_model_lite_file)

            interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_dir_lite+current_model_lite_file)

            test(name, interpreter, test_images, labels)
    else:
        print("Model not found in {}".format(config.model_file))


















