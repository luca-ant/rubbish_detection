import wget
import zipfile
import os
import shutil
import random
import re
import config

def download_dataset(working_dir='./'):
    data_dir = working_dir+'data/'
    dataset_dir = working_dir+'data/dataset/'
    if not os.path.exists(dataset_dir):

            os.makedirs(data_dir, exist_ok=True)

            print("DOWNLOADING DATASET")
            os.system("git clone --progress -v https://github.com/luca-ant/rubbish_dataset.git " + data_dir )

    else:
            print("Dataset already exists")


if __name__ == "__main__":
    download_dataset(working_dir=config.working_dir)
