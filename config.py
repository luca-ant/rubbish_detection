working_dir="./"
model_dir = working_dir + "model/"
model_dir_lite = working_dir + "models_lite/"
test_res_dir_lite = working_dir + "test_results_conv_lite/"
weights_dir = working_dir + "weights/"
dataset_dir = working_dir+'data/dataset/'
train_dir = dataset_dir+'train/'
test_dir = dataset_dir+'test/'
val_dir = dataset_dir+'val/'


weights_file = weights_dir + "rd_weights.h5"
model_file = model_dir + "rd_model.h5"
model_checkpoint = model_dir + "rd_checkpoint.h5"
model_lite_file = model_dir_lite + "rd_model_lite.tflite"
labels_file = dataset_dir+'labels.txt'

input_shape=(224,224,3)
#input_shape=(299,299,3)
batch_size = 16
total_epochs = 100
