import tensorflow as tf
import numpy as np
import cv2
import config
import os
from predict_rubbish_lite import predict_class_lite
from preprocess_data import load_labels


height = config.input_shape[0]
width = config.input_shape[1]

cam = cv2.VideoCapture(0)
bgs = cv2.createBackgroundSubtractorMOG2()

labels = load_labels(config.labels_file)

if os.path.isfile(config.model_lite_file):
    interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_lite_file)
else:
    print("Model lite not found in {}".format(config.model_lite_file))
    exit(1)

while(1):

    for i in range(5):
        cam.grab()
    ret, frame = cam.read()

    cv2.imshow('frame',frame)

    k = cv2.waitKey(5) & 0xff
    if k == 27: # ESC
        break

    if k == 32: #SPACE
        img = cv2.resize(frame, (width, height))
        predictions = predict_class_lite(interpreter, img ,labels)

        print(' '+'='*21+' ')
        for k, v in predictions.items():
            print('|{:12}| {:6.2f}%|'.format(k, v*100))
        print(' '+'='*21)


cam.release()
cv2.destroyAllWindows()
