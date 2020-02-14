import wget
import zipfile
import os
import shutil
import random
from collections import defaultdict
import re

class Dataset():

    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.data_dir = working_dir+'data/'
        self.dataset_dir = working_dir+'data/dataset/'
        self.dataset = {}
        self.train_images = []
        self.test_images = []
        self.val_images = []
        self.num_classes = -1
    def download_dataset(self):
        if not os.path.exists(self.dataset_dir):

                os.makedirs(self.data_dir, exist_ok=True)

                print("DOWNLOADING DATASET")
                shutil.rmtree(self.working_dir+'data', ignore_errors=True)

                os.system("git clone --progress -v https://github.com/luca-ant/rubbish_dataset.git " + self.data_dir )

        else:
                print("Dataset already exists")

    def load_dataset(self):

        self.download_dataset()
        self.dataset = defaultdict(list)

        with os.scandir(self.dataset_dir) as entries:
            for e in entries:
                if e.is_file():
                    c = re.split(r'[0-9]', e.name)[0]
                    self.dataset[c].append(e.name)

        self.num_classes = len(self.dataset.keys())
        return self.dataset
        

    def split_train_test_val(self):
        #for c in self.dataset.keys():
        #    random.shuffle(self.dataset[c])
        
        for c in self.dataset.keys():
            self.train_images = self.train_images + self.dataset[c][int(len(self.dataset[c]) * 0.0) : int(len(self.dataset[c]) * 0.50)]
            self.test_images = self.test_images + self.dataset[c][int(len(self.dataset[c]) * 0.50) : int(len(self.dataset[c]) * 0.75)]
            self.val_images = self.val_images + self.dataset[c][int(len(self.dataset[c]) * 0.75) : int(len(self.dataset[c]) * 1.0)]


        random.shuffle(self.train_images)
        random.shuffle(self.test_images)
        random.shuffle(self.val_images)

        return self.train_images, self.test_images, self.val_images

