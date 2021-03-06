import tensorflow as tf
import numpy as np
import cv2
import config
import re
import os
import time 
from progress.bar import Bar
from predict_rubbish_lite import predict_class_lite
from preprocess_data import decode_label, load_labels, load_test_dataset, read_image_as_array


def test(name, interpreter, test_images, labesl):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('intput ', input_details)
    print('output', output_details)
    target_shape = input_details[0]['shape'][1:]

    accurate_count = 0
    total_time = 0

    bar = Bar('Testing tflite model '+ name, max=len(test_images))

    for image_name in test_images:
        true_label = re.split(r'[0-9]', image_name)[0]
        image_array = read_image_as_array(config.test_dir+true_label+'/'+image_name, target_shape=target_shape)
        if 'int-quant' in name:
            image_array = image_array * 255.
            image_array = image_array.astype(np.uint8)

        img_batch = np.expand_dims(image_array, 0)

        interpreter.set_tensor(input_details[0]['index'], img_batch)
        
        start = time.time()
        interpreter.invoke()
        stop = time.time()
        total_time += stop - start

        output_data = interpreter.get_tensor(output_details[0]['index'])

        if 'int-quant' in name:
            output_data = output_data.astype(np.float32)/255.
#       predictions = dict(zip(labels, list(output_data[0])))
#       predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
#       print(' '+'='*21+' ')
#       for k, v in predictions.items():
#           print('|{:12}| {:6.2f}%|'.format(k, v*100))
#        print(' '+'='*21)
        
        if true_label.strip() == decode_label(labels, output_data).strip():
            accurate_count += 1
        bar.next()
    bar.finish()

    accuracy = accurate_count * 1.0 / len(test_images)
    total_time = total_time * 1000
    print('\nMODEL: {}'.format(model_name))
    print('ACCURACY: {:.2f}'.format(accuracy *100))
    print('LATENCY: {:.9f}\n'.format(total_time /len(test_images)))

    os.makedirs(config.test_res_tflite_dir, exist_ok=True)
    model = model_name.split('_')[0]
    with open(config.test_res_tflite_dir+model +'.csv', "a") as f:
        f.write('{};{:.2f};{:.9}\n'.format(model_name, accuracy*100,total_time/len(test_images)))

    with open(config.test_res_tflite_file, "a") as f:
        f.write('{};{:.2f};{:.9}\n'.format(model_name, accuracy*100,total_time/len(test_images)))


if __name__ == "__main__":

    labels = load_labels(config.labels_file)
    test_images = load_test_dataset(config.test_dir)


    if os.path.isdir(config.models_tflite_dir):
        os.makedirs(config.test_res_tflite_dir, exist_ok=True)

        with open(config.test_res_tflite_file, "w") as f:
            f.write('{};{};{}\n'.format("Model name", "Accuracy (%)", "Latency (ms)"))
        with os.scandir(config.models_tflite_dir) as entries:

            for e in entries:
                if e.is_file():
                    model_tflite_file = e.name
                    model_name = ''.join(model_tflite_file.split('.')[0])
                    model = model_name.split('_')[0]
                with open(config.test_res_tflite_dir + model+'.csv', "w") as f:
                    f.write('{};{};{}\n'.format("Model name", "Accuracy (%)", "Latency (ms)"))

        with os.scandir(config.models_tflite_dir) as entries:

            for e in entries:
                if e.is_file():
                    model_tflite_file = e.name
                    model_name = ''.join(model_tflite_file.split('.')[0])

                    interpreter = tf.compat.v2.lite.Interpreter(model_path=config.models_tflite_dir + model_tflite_file)

                    test(model_name, interpreter, test_images, labels)
    else:
        print("Models lite not found in {}".format(config.models_tflite_dir))


















