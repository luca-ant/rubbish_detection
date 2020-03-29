# Rubbish detector

## Getting started

* Clone repository
```
git clone https://github.com/luca-ant/rubbish_detector.git
```

* Install dependencies
```
sudo apt install python3-setuptools
sudo apt install python3-pip
sudo apt install python3-venv
```

* Create a virtual environment and install requirements modules
```
cd rubbish_detector
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -r requirements.txt
```

## Download dataset
Run the script ```download_dataset.py``` to download the dataset in the correct structure.

```
python download_dataset.py
```


### Download already trained keras models
Download the zip archive containing all models and extract it in the main directory fo the repository. Use the following commands:

```
cd rubbish_detection
wget https://github.com/luca-ant/rubbish_detection/releases/download/models/models.zip
tar -xvf models_tflite.tgz
```


## Configuration

Change the file ```config.py``` and uncomment the model that you prefer.

## Running

* **Training:** After choose the model in the config.py file, run the script ```train_rubbish_detector.py```. You'll find training history and data in ```training_logs``` directory.

```
python train_rubbish_detector.py
```

* **Evaluate:** To evaluate whole model on test images and calculate accuracy run the script ```evaluate_rubbish_detector.py``` 

```
python evaluate_rubbish_detector.py
```


* **Predict:** To use the classifier to predict a class for your image run the script ```predict_rubbish.py```

```
python predict_rubbish.py PATH_TO_YOUR_IMAGE 
```

To use the webcam of your pc as input use the script ```camera.py``` and press ```SPACE BAR``` to take a photo and predict the the class.

```
python camera.py
```


## Converting to Tensorflow Lite


To convert the model into tflite version run the script ```convert_keras_to_tflite.py```

```
python convert_keras_to_tflite.py
```


* **Evaluate:** To evaluate the non optimized tflite model run the script ```evaluate_rubbish_detector_lite.py``` 

```
python evaluate_rubbish_detector_lite.py
```


* **Predict:** To use the non optimized tflite model to predict a class for your image run the script ```predict_rubbish.py```

```
python predict_rubbish_lite.py PATH_TO_YOUR_IMAGE 
```



* **Test:** To test and compute accuracy of all optimized tflite models run the script ```test_tflite.py```. You'll find the output in ```test_results_tflite``` directory.

```
python test_tflite.py
```


### Download already converted and optimized tflite models
Download the tar archive containing all tflite models and extract it in the main directory fo the repository. Use the following commands:

```
cd rubbish_detection
wget https://github.com/luca-ant/rubbish_detection/releases/download/models_tflite/models_tflite.tgz
tar -xvf models_tflite.tgz
```

## Results


| Model | Accuracy |
| :---          |  :---: |
| InceptionV3   | 94.95% |
| MobileNetV2   | 92.55% |
| NASNetMobile  | 93.62% |
| ResNet50      | 94.68% |


