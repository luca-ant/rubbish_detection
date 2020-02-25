import os
import numpy as np
import config
import rubbish_detector_model
from tensorflow.python.keras.preprocessing import image
from preprocess_data import decode_label, load_labels
from preprocess_data import data_generator, load_labels, load_train_dataset, load_test_dataset, load_val_dataset


def predict_class(model, image_name, labels):

        img = image.load_img(image_name, target_size=config.input_shape)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        #print(img.shape)
        #print(model.layers[0].layers[0].input_shape)

        output_data = model.predict(x=img, verbose=1,)

        predictions = dict(zip(labels, list(output_data[0])))
        predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        print(' '+'='*21+' ')
        for k, v in predictions.items():
            print('|{:12}| {:6.2f}%|'.format(k, v*100))
        print(' '+'='*21)
        print("\nLABEL", decode_label(labels, output_data))




if __name__ == "__main__":
    
    image_name = '/home/luca/Desktop/rubbish_detection/data/dataset/test/metal2.jpg'

    labels = load_labels(config.labels_file)
  
    if os.path.isdir(config.model_dir):
        model = rubbish_detector_model.restore_model(config.model_file, config.weights_file, len(labels))

        predict_class(model, image_name, labels)
    else:
        print("Model not found in {}".format(config.model_dir))
