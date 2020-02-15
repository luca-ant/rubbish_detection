
import config
from tensorflow.python.keras.preprocessing import image

def predict_class(model, image_name)

        img = image.load_img(image_name, target_size=(224, 224, 3))
        img = image.img_to_array(img)




if __name__ == "__main__":
    
    image_name = ''

    if os.path.isdir(config.model_dir):
        model = rubbish_detector_model.restore_model(config.model_file)

    predict(model, image_name)