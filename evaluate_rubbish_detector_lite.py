import tensorflow as tf
import cv2
import numpy as np
import os
import re
import traceback
import config
from tensorflow.python.keras.preprocessing import image
from preprocess_data import decode_label, load_labels, load_test_dataset

height = config.input_shape[0]
width = config.input_shape[1]


def evaluate(interpreter, labels, test_images):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    print('intput ', input_details)
    print('output', output_details)

    accurate_count = 0

    for image_name in test_images:
        true_label = re.split(r'[0-9]', image_name)[0]

        img = image.load_img(config.test_dir+image_name, target_size=config.input_shape)
        img = image.img_to_array(img)


#        img_o = cv2.imread(image_name)
#        img = cv2.resize(img_o, (width, height))
#        cv2.imshow('image', img)
#        k = cv2.waitKey(0)

        img_batch = np.expand_dims(img, 0)

#        img_batch = np.float32(img_batch)

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

    accuracy = accurate_count * 1.0 / len(test_images)
    print('\nACCURACY: {:.2f}%'.format(accuracy *100))
    return accuracy


if __name__ == "__main__":


    labels = load_labels(config.labels_file)
    test_images = load_test_dataset(config.test_dir)


    if os.path.isfile(config.model_lite_file):
        interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_lite_file)
        interpreter.allocate_tensors()
        evaluate(interpreter, labels, test_images)
    else:
        print("Model not found in {}".format(config.model_file_lite))
