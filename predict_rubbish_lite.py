import tensorflow as tf
import os
import sys
import cv2
import numpy as np
import time
import traceback
import config
from PIL import Image
from tensorflow.python.keras.preprocessing import image
from preprocess_data import load_labels, decode_label, read_image_as_array



def predict_class_lite(interpreter, image_array, labels):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_array = np.float32(image_array)
    image_array = image_array / .255

    if 'full-int' in config.model_tflite_file:
        image_array = image_array.astype(np.uint8)

    img_batch = np.expand_dims(image_array, 0)


    interpreter.set_tensor(input_details[0]['index'], img_batch)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    if 'full-int' in config.model_tflite_file:
        output_data = output_data.astype(np.float32)/255.

    print('output_data:', output_data[0], "type:", type(output_data[0][0]))
    predictions = dict(zip(labels, list(output_data[0])))
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    return predictions 






if __name__ == "__main__":
    
    #image_path = '/home/luca/Desktop/rubbish_detection/data/dataset/test/glass/glass405.jpg'
    if len(sys.argv) != 2:
        print("Usage: {} PATH_TO_YOUR_IMAGE".format(sys.argv[0]))
        exit(1)
    image_path = sys.argv[1]
    image_array = read_image_as_array(image_path)
    Image.fromarray(np.uint8(image_array*255)).show()
    labels = load_labels(config.labels_file)

    if os.path.isfile(config.model_tflite_file):
        interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_tflite_file)
        interpreter.allocate_tensors()
        predictions = predict_class_lite(interpreter, image_array,labels)

        print(' '+'='*21+' ')
        for k, v in predictions.items():
            print('|{:12}| {:6.2f}%|'.format(k, v*100))
        print(' '+'='*21)
        predicted_label = list(predictions.keys())[0]
        print("\nLABEL", predicted_label)
    else:
        print("Model lite not found in {}".format(config.model_tflite_file))




