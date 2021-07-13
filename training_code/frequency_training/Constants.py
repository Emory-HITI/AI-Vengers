from pathlib import Path
import pandas as pd
import numpy as np
import os

## CXR

df_paths = {
    'MIMIC': {
        'train': "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/laleh/new_split/8-1-1/new_train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/laleh/new_split/8-1-1/new_valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/laleh/new_split/8-1-1/new_test.csv"        
    },
    'CXP':{
        'train': "/scratch/hdd001/projects/ml4h/projects/CheXpert/split/July19/new_train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/CheXpert/split/July19/new_valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/CheXpert/split/July19/new_test.csv"
    },
    'NIH':{
        'train': "/scratch/hdd001/projects/ml4h/projects/NIH/split/July16/train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/NIH/split/July16/valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/NIH/split/July16/test.csv"
    },
    'PAD':{
        'train': "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/haoran_split/train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/haoran_split/valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/haoran_split/test.csv"            
    }
}

image_paths = {
    'MIMIC':  "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/",
    'CXP': "/scratch/hdd001/projects/ml4h/projects/CheXpert/",
    'NIH': "/scratch/hdd001/projects/ml4h/projects/NIH/images/",
    'PAD': '/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/images-224'
}

MIMIC_details = "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/vin_new_split/8-1-1/mimic-cxr-metadata-detail.csv"
CXP_details = "/scratch/hdd001/projects/ml4h/projects/CheXpert/chexpert_demographics.csv"
PAD_details = "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
cache_dir = "/scratch/hdd001/home/{}/datasets/cache".format(os.environ.get('USER'))

IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

take_labels = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema']

race_mapping = {
 0: 'White',
 1: "Black",
 2: "Hispanic",
 3: "Asian",
 4: "Other"
}

gender_mapping = {
    0: 'F',
    1: 'M'
}

N_PATCHES = 9
