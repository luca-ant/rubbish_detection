import os
import numpy as np
import config
import rubbish_detector_model
from tensorflow.python.keras.preprocessing import image
from preprocess_data import decode_label, load_labels
from preprocess_data import data_generator, load_labels, load_train_dataset, load_test_dataset, load_val_dataset


def predict_class(model, image_name, labels):

        img = image.load_img(image_name, target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        #print(img.shape)
        #print(model.layers[0].layers[0].input_shape)

        label = model.predict(x=img, batch_size=1, verbose=1,)

        print(decode_label(labels, label))





if __name__ == "__main__":
    
    image_name = '/home/luca/Desktop/rubbish_detection/data/dataset/test/metal209.jpg'

    labels = load_labels(config.labels_file)
  
    if os.path.isdir(config.model_dir):
        model = rubbish_detector_model.restore_model(config.model_file, config.weights_file, len(labels))

    predict_class(model, image_name, labels)
