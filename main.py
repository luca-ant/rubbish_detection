from preprocess_data import *
from train_rubbish_detector import *
import shutil

dataset, num_classes = load_dataset()
train_images, test_images, val_images = split_train_test_val(dataset)



for i in train_images:
    shutil.move(config.dataset_dir+i, config.dataset_dir+'train/'+i)

for i in test_images:
    shutil.move(config.dataset_dir+i, config.dataset_dir+'test/'+i)

for i in val_images:
    shutil.move(config.dataset_dir+i, config.dataset_dir+'val/'+i)

#encode_label(list(dataset.keys()), 'metal')
#train_data_generator = data_generator("./data/dataset/", list(dataset.keys()), train_images, batch_size)
