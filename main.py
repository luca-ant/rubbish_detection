import rubbish_detector_model 
import config
from preprocess_data import data_generator, load_labels, load_train_dataset, load_val_dataset
import shutil



labels = load_labels(config.labels_file)
#model = rubbish_detector_model.create_nn(len(labels))

model = rubbish_detector_model.restore_model(config.model_file, config.weights_file, len(labels))

#print("SAVING MODEL TO " + config.model_file)

#model.save(config.model_file, include_optimizer=False)
