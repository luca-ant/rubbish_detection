##### RESNET50 #####
#model_name = 'resnet50'
#input_shape=(224,224,3)
####################

##### INCEPTION_V3 #####
#model_name = 'inceptionV3'
#input_shape=(299,299,3)
########################

##### MOBILENET_V2 #####
model_name = 'mobilenetV2'
input_shape=(224,224,3)
########################

####### VGG19 #######
#model_name = 'vgg19'
#input_shape=(224,224,3)
#####################

##### NASNETMOBILE #####
#model_name = 'nasnetmobile'
#input_shape=(224,224,3)
########################


working_dir="./"
models_dir = working_dir + "models/"
models_tflite_dir = working_dir + "models_tflite/"
test_res_dir_tflite = working_dir + "test_results_tflite/"
train_log_dir = working_dir + "training_logs/"

dataset_dir = working_dir+'data/dataset/'
train_dir = dataset_dir+'train/'
test_dir = dataset_dir+'test/'
val_dir = dataset_dir+'val/'
labels_file = dataset_dir+'labels.txt'


model_file = models_dir + model_name+".h5"
model_checkpoint = models_dir + model_name+"_checkpoint.h5"
model_tflite_file = models_tflite_dir + model_name+".tflite"
train_log_file = train_log_dir + model_name+"_log.csv"

batch_size = 16
total_epochs = 100
