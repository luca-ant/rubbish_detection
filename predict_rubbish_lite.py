import tensorflow as tf
import os
import cv2
import numpy as np
import time
import traceback
import config
from PIL import Image
from tensorflow.python.keras.preprocessing import image
from preprocess_data import load_labels, decode_label, read_image_as_array


height = config.input_shape[0]
width = config.input_shape[1]


def predict_class_lite(interpreter, image_array, labels):

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    print('intput ', input_details)
    print('output', output_details)



    img_batch = np.expand_dims(image_array, 0)


    img_batch = np.float32(img_batch)

    interpreter.set_tensor(input_details[0]['index'], img_batch)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('output_data', output_data[0])
    predictions = dict(zip(labels, list(output_data[0])))
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    return predictions 






if __name__ == "__main__":
    
    image_name = '/home/luca/Desktop/rubbish_detection/data/dataset/test/glass405.jpg'

    image_array = read_image_as_array(image_name)
    Image.fromarray(np.uint8(image_array)).show()
    labels = load_labels(config.labels_file)

    if os.path.isfile(config.model_lite_file):
        interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_lite_file)
        predictions = predict_class_lite(interpreter, image_array,labels)

        print(' '+'='*21+' ')
        for k, v in predictions.items():
            print('|{:12}| {:6.2f}%|'.format(k, v*100))
        print(' '+'='*21)
        predicted_label = list(predictions.keys())[0]
        print("\nLABEL", predicted_label)
    else:
        print("Model lite not found in {}".format(config.model_lite_file))




