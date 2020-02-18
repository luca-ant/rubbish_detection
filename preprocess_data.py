import re
import numpy as np
import os
import random
from collections import defaultdict
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical



def load_labels(working_dir='./'):

    dataset_dir = working_dir+'data/dataset/'
    labels_file = dataset_dir+'labels.txt'
    labels = []

    with open(labels_file) as lf:
        for l in lf:
            labels.append(l.strip())
    labels.sort()
    return labels

def load_train_dataset(working_dir='./'):

    dataset_dir = working_dir+'data/dataset/'
    train_dir =dataset_dir+'train/'
    train_images = []

    with os.scandir(train_dir) as entries:
        for e in entries:
            if e.is_file():
                train_images.append(e.name)
    
    return train_images


def load_test_dataset(working_dir='./'):

    dataset_dir = working_dir+'data/dataset/'
    test_dir =dataset_dir+'test/'
    test_images = []

    with os.scandir(test_dir) as entries:
        for e in entries:
            if e.is_file():
                test_images.append(e.name)
    
    return test_images


def load_val_dataset(working_dir='./'):

    dataset_dir = working_dir+'data/dataset/'
    train_dir =dataset_dir+'val/'
    val_images = []

    with os.scandir(test_dir) as entries:
        for e in entries:
            if e.is_file():
                val_images.append(e.name)
    
    return val_images



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
            img = image.load_img(dataset_dir + image_name, target_size=(224, 224, 3))
            img = image.img_to_array(img)
            c = re.split(r'[0-9]', image_name)[0]

            encoded_label = to_categorical(labels_int_dict[c], num_classes=len(labels))
            x_image.append(img)
            y_class.append(encoded_label)

            if n == bath_size:
                yield (np.array(x_image),  np.array(y_class))
                x_image, y_class = list(), list()
                n = 0