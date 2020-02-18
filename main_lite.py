import tensorflow as tf
import cv2
import numpy as np
import time
import traceback
import config
from preprocess_data import load_labels, decode_label

DEPTH_FACTOR = 15
DISP_FACTOR = 6

labels = load_labels(config.working_dir)
interpreter = tf.compat.v1.lite.Interpreter(model_path='./model_lite/model_lite.tflite')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print('intput ', input_details)
print('output', output_details)

height = 224
width = 224

image_name = '/home/luca/rubbish_detection/data/dataset/test/metal209.jpg'

img = cv2.imread(image_name)
# Prepare input to the network
img = cv2.resize(img, (width, height)).astype(np.float32)
img_batch = np.expand_dims(img, 0)
print('img ', img.shape)
interpreter.allocate_tensors()

#   img_batch = np.float32(img_batch)

print("image_batch", img_batch.shape)
print ('Set tensor index ', input_details[0]['index'])
print(input_details[0])

print(interpreter.get_tensor_details()[:4])

interpreter.set_tensor(input_details[0]['index'], img_batch)

print("input tensor set")


interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

print("LABEL", decode_label(labels, output_data))

print("FINEEEE")

img = (img*255).astype(np.uint8)            


while True:
    # Build final output

    # Show cool visualization
    cv2.imshow('aa', img)
