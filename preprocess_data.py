import re
import numpy as np
from tensorflow.python.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input




def data_generator(dataset_dir, dataset_list, bath_size):
    
    x_image, y_class = list(), list()
    n = 0
    while True:

        for image_name in dataset_list:
            n += 1


            img = image.load_img(dataset_dir + image_name, target_size=(224, 224, 3))
            img = image.img_to_array(img)
            c = re.split(r'[0-9]', image_name)[0]

            x_image.append(img)
            y_class.append(c)

            if n == bath_size:
                yield [np.array(x_image),  np.array(y_class)]
                x_image, y_class = list(), list()
                n = 0