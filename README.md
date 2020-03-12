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

## Configuration

Change the file ```config.py``` and uncomment the model that you prefer.

## Running

* **Training:** After choose the model in the config.py file, run the script train_rubbish_detector.py 

```
python train_rubbish_detector.py
```

* **Evaluate:** To evaluate whole model on test images and calculate accuracy run the script ```evaluate_rubbish_detector.py``` 

```
python evaluate_rubbish_detector.py
```


* **Predict:** To use tha classifier to predict a a class for your image. Use the script ```predict_rubbish.py```

```
python predict_rubbish.py PATH_TO_YOUR_IMAGE 
```


## Converting to Tensorflow Lite




## Results


### Examples


