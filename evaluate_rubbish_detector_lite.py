import tensorflow as tf
import cv2
import numpy as np
import os
import re
import traceback
import config
from progress.bar import Bar
from tensorflow.python.keras.preprocessing import image
from preprocess_data import decode_label, load_labels, load_test_dataset, read_image_as_array



def evaluate(interpreter, labels, test_images):

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    print('intput ', input_details)
    print('output', output_details)

    accurate_count = 0

    bar = Bar('Evaluating images', max=len(test_images))
    for image_name in test_images:
        true_label = re.split(r'[0-9]', image_name)[0]
        image_array = read_image_as_array(config.test_dir+true_label+'/'+image_name)

        img_batch = np.expand_dims(image_array, 0)


        interpreter.set_tensor(input_details[0]['index'], img_batch)


        interpreter.invoke()

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
    return accuracy


if __name__ == "__main__":


    labels = load_labels(config.labels_file)
    test_images = load_test_dataset(config.test_dir)


    if os.path.isfile(config.model_tflite_file):
        interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_tflite_file)
        evaluate(interpreter, labels, test_images)
    else:
        print("Model not found in {}".format(config.model_tffile_lite))
