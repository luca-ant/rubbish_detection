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

labels = load_labels(config.labels_file)

#bgs = cv2.createBackgroundSubtractorMOG2()
#bgs = cv2.createBackgroundSubtractorKNN()

if os.path.isfile(config.model_tflite_file):
    interpreter = tf.compat.v2.lite.Interpreter(model_path=config.model_tflite_file)
else:
    print("Model lite not found in {}".format(config.model_tflite_file))
    exit(1)


while(1):

#    for i in range(5):
#        cam.grab()
    _, frame = cam.read()

    
    cv2.imshow('Frame', frame)
    
    frame = cv2.GaussianBlur(frame,(5,5),0)

#########################
#    fgMask = bgs.apply(frame)
#    cv2.imshow('Mask', fgMask)
#    mask = np.stack((fgMask,fgMask, fgMask), axis=2)
#    image = frame * mask
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    cv2.imshow('Detect', image)
#########################

    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC
        break

    if k == 32: #SPACE
        img = frame 
        img = cv2.resize(img, (width, height))
        predictions = predict_class_lite(interpreter, img ,labels)

        print(' '+'='*21+' ')
        for k, v in predictions.items():
            print('|{:12}| {:6.2f}%|'.format(k, v*100))
        print(' '+'='*21)


cam.release()
cv2.destroyAllWindows()
