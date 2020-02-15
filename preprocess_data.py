import re
import numpy as np
import os
import random
from collections import defaultdict
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical


def load_dataset(working_dir='./'):

    working_dir = './'
    data_dir = working_dir+'data/'
    dataset_dir = working_dir+'data/dataset/'
    if not os.path.exists(dataset_dir):
        print('Dataset NOT FOUND!')
        return 

    dataset = defaultdict(list)

    with os.scandir(dataset_dir) as entries:
        for e in entries:
            if e.is_file():
                c = re.split(r'[0-9]', e.name)[0]
                dataset[c].append(e.name)

    labels = list(dataset.keys())

    return dataset, labels
    

def split_train_test_val(dataset):
    for c in dataset.keys():
        dataset[c].sort()
    #    random.shuffle(dataset[c])
    train_images = []
    test_images = []
    val_images = []
    for c in dataset.keys():
        train_images = train_images + dataset[c][int(len(dataset[c]) * 0.0) : int(len(dataset[c]) * 0.50)]
        test_images = test_images + dataset[c][int(len(dataset[c]) * 0.50) : int(len(dataset[c]) * 0.75)]
        val_images = val_images + dataset[c][int(len(dataset[c]) * 0.75) : int(len(dataset[c]) * 1.0)]


    random.shuffle(train_images)
    random.shuffle(test_images)
    random.shuffle(val_images)

    return train_images, test_images, val_images



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