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


def test(name, interpreter, test_images, labesl):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accurate_count = 0
    total_time = 0

    bar = Bar('Testing tflite model '+ name, max=len(test_images))

    for image_name in test_images:
        true_label = re.split(r'[0-9]', image_name)[0]
        image_array = read_image_as_array(config.test_dir+image_name)
        if 'full-int' in name:
            image_array = image_array.astype(np.uint8)
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
    print('TIME/IMAGE: {:.6f} sec\n'.format(total_time /len(test_images)))

    os.makedirs(config.test_res_dir_lite, exist_ok=True)
    with open(config.test_res_dir_lite+name+'_results.txt', "w") as f:
        f.write('ACCURACY: {:.2f}%\n'.format(accuracy *100))
        f.write('TIME/IMAGE: {:.6} sec'.format(total_time /len(test_images)))



if __name__ == "__main__":

    labels = load_labels(config.labels_file)
    test_images = load_test_dataset(config.test_dir)


    if os.path.isdir(config.models_tflite_dir):

        with os.scandir(config.models_tflite_dir) as entries:

            for e in entries:
                if e.is_file():
                    model_tflite_file = e.name
                    model_name = ''.join(model_tflite_file.split('.')[0])

                    interpreter = tf.compat.v2.lite.Interpreter(model_path=config.models_dir_tflite + model_tflite_file)

                    test(model_name, interpreter, test_images, labels)
    else:
        print("Models lite not found in {}".format(config.models_dir_tflite))


















