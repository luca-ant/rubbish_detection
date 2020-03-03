import os
import numpy as np
import config
import rubbish_detector_model
from PIL import Image
from tensorflow.python.keras.preprocessing import image
from preprocess_data import decode_label, load_labels, read_image_as_array
from preprocess_data import data_generator, load_labels, load_train_dataset, load_test_dataset, load_val_dataset


def predict_class(model, image_array, labels):

    img_batch = np.expand_dims(image_array, 0)

    output_data = model.predict(x=img_batch, verbose=1,)
#    print('output_data', output_data[0])

    predictions = dict(zip(labels, list(output_data[0])))
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

    return predictions


if __name__ == "__main__":
    
    image_path = '/home/luca/Desktop/rubbish_detection/data/dataset/test/glass/glass405.jpg'
    image_array = read_image_as_array(image_path)
    Image.fromarray(np.uint8(image_array)).show()
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
