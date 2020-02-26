import re
import numpy as np
import os
import random
import config
import cv2
from collections import defaultdict
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical


def load_labels(labels_file):

    labels = []

    with open(labels_file) as lf:
        for l in lf:
            labels.append(l.strip())
    labels.sort()
    return labels


def read_image_as_array(image_name):

    height = config.input_shape[0]
    width = config.input_shape[1]

    # load with Pillow
    img = image.load_img(image_name, target_size=config.input_shape)
#    img.show()
    image_array = image.img_to_array(img)

    # load with opencv
#    img_o = cv2.imread(image_name)
#    img_o = cv2.resize(img_o, (width, height))
#    image_array = cv2.cvtColor(img_o,cv2.COLOR_BGR2RGB)
#    cv2.imshow('image', img_o)
#    k = cv2.waitKey(0)

    return image_array







def load_dataset_from_dir(d):

    images = []

    with os.scandir(d) as entries:
        for e in entries:
            if e.is_file():
                images.append(e.name)
    random.shuffle(images) 
    return images


def load_train_dataset(train_dir):
    
    return load_dataset_from_dir(train_dir)


def load_test_dataset(test_dir):
    return load_dataset_from_dir(test_dir)

def load_val_dataset(val_dir):

    return load_dataset_from_dir(val_dir)



# def split_train_test_val(dataset):
#     for c in dataset.keys():
#         random.shuffle(dataset[c])
#     train_images = []
#     test_images = []
#     val_images = []
#     for c in dataset.keys():
#         train_images = train_images + dataset[c][int(len(dataset[c]) * 0.0) : int(len(dataset[c]) * 0.7)]
#         test_images = test_images + dataset[c][int(len(dataset[c]) * 0.7) : int(len(dataset[c]) * 0.85)]
#         val_images = val_images + dataset[c][int(len(dataset[c]) * 0.85) : int(len(dataset[c]) * 1.0)]


#     random.shuffle(train_images)
#     random.shuffle(test_images)
#     random.shuffle(val_images)

#     return train_images, test_images, val_images



def decode_label(labels, label):
    labels.sort()
    return labels[np.argmax(label)]


def data_generator(dataset_dir, labels, dataset_list, bath_size):
    x_image, y_class = list(), list()
    labels.sort()
    labels_int_dict = {labels[i]: i for i in range(0, len(labels))}

    n = 0
    while True:

        for image_name in dataset_list:
            n += 1
#            img = image.load_img(config.test_dir + image_name, target_size=config.input_shape)
#            img = image.img_to_array(img)
            image_array = read_image_as_array(dataset_dir + image_name)
            c = re.split(r'[0-9]', image_name)[0]

            encoded_label = to_categorical(labels_int_dict[c], num_classes=len(labels))
            x_image.append(image_array)
            y_class.append(encoded_label)

            if n == bath_size:
                yield (np.array(x_image),  np.array(y_class))
                x_image, y_class = list(), list()
                n = 0
