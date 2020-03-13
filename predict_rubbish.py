import os
import sys
import numpy as np
import config
import rubbish_detector_model
from PIL import Image
from tensorflow.python.keras.preprocessing import image
from preprocess_data import decode_label, load_labels, read_image_as_array
from preprocess_data import data_generator, load_labels, load_train_dataset, load_test_dataset, load_val_dataset


def predict_class(model, image_array, labels):

    image_array = np.float32(image_array)
    image_array = image_array / .255
    img_batch = np.expand_dims(image_array, 0)

    output_data = model.predict(x=img_batch, verbose=1,)
    print('output_data:', output_data[0], "type:", type(output_data[0][0]))

    predictions = dict(zip(labels, list(output_data[0])))
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

    return predictions


if __name__ == "__main__":
     
    #image_path = '/home/luca/Desktop/rubbish_detection/data/dataset/test/glass/glass405.jpg'
    if len(sys.argv) != 2:
        print("Usage: {} PATH_TO_YOUR_IMAGE".format(sys.argv[0]))
        exit(1)
    image_path = sys.argv[1]
    image_array = read_image_as_array(image_path)
    Image.fromarray(np.uint8(image_array*255)).show()
    labels = load_labels(config.labels_file)
  
    if os.path.isfile(config.model_file):
        model = rubbish_detector_model.restore_model(config.model_file)
        predictions = predict_class(model, image_array, labels)
        print(' '+'='*21+' ')
        for k, v in predictions.items():
            print('|{:12}| {:6.2f}%|'.format(k, v*100))
        print(' '+'='*21)
        predicted_label = list(predictions.keys())[0]
        print("\nLABEL", predicted_label)
    else:
        print("Model not found in {}".format(config.model_file))
