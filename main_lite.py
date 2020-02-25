import tensorflow as tf
import cv2
import numpy as np
import time
import traceback
import config
from preprocess_data import load_labels, decode_label

DEPTH_FACTOR = 15
DISP_FACTOR = 6

labels = load_labels(config.labels_file)
interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_lite_file)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print('intput ', input_details)
print('output', output_details)

height = config.input_shape[0]
width = config.input_shape[1]

image_name = '/home/luca/Desktop/rubbish_detection/data/dataset/test/metal2.jpg'

img_o = cv2.imread(image_name)

# Prepare input to the network
img = cv2.resize(img_o, (width, height))

cv2.imshow('image', img)
k = cv2.waitKey(0)

img_batch = np.expand_dims(img, 0)

print('img shape', img.shape)

interpreter.allocate_tensors()

img_batch = np.float32(img_batch)

print("image_batch", img_batch.shape)
print ('Set tensor index ', input_details[0]['index'])
print(input_details[0])

print(interpreter.get_tensor_details()[:4])

interpreter.set_tensor(input_details[0]['index'], img_batch)

print("input tensor set")


interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

predictions = dict(zip(labels, list(output_data[0])))
predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
print(' '+'='*21+' ')
for k, v in predictions.items():
    print('|{:12}| {:6.2f}%|'.format(k, v*100))
print(' '+'='*21)
print("\nLABEL", decode_label(labels, output_data))

