import Constants
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
import re

def preprocess_MIMIC(split, only_frontal, return_all_labels = False):
    details = pd.read_csv(Constants.MIMIC_details)
    details = details.drop(columns=['dicom_id', 'study_id', 'religion', 'marital_status', 'gender'])
    details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
    df = pd.merge(split, details)

    copy_subjectid = df['subject_id']
    df = df.drop(columns = ['subject_id']).replace(
            [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
             'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
             '>=90'],
            [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
             'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])

    df['subject_id'] = copy_subjectid.astype(str)
    df['Age'] = df["age_decile"]
    df['Sex'] = df["gender"]
    df = df.drop(columns=["age_decile", 'gender'])
    df = df.rename(
        columns = {
            'Pleural Effusion':'Effusion',
        })
    df['study_id'] = df['path'].apply(lambda x: x[x.index('p'):x.rindex('/')])
    df['path'] = Constants.image_paths['MIMIC'] + df['path'].astype(str)
    df['frontal'] = (df.view == 'frontal')
    if only_frontal:
        df = df[df.frontal]

    df['env'] = 'MIMIC'
    df.loc[df.Age == 0, 'Age'] = '0-20'

    df = df[(~df.race.isin(['UNKNOWN', 'UNABLE TO OBTAIN', 0, '0'])) & (~pd.isnull(df.race))]

    race_mapping = defaultdict(lambda:4)
    race_mapping['WHITE'] = 0
    race_mapping['BLACK/AFRICAN AMERICAN'] = 1
    race_mapping['HISPANIC/LATINO'] = 2
    race_mapping['ASIAN'] = 3

    df['race'] = df['race'].map(race_mapping)
    df['Sex'] = (df['Sex'] == 'M').astype(int)

    return df[['subject_id','path','Sex', "Age", 'env', 'frontal', 'study_id', 'race'] + Constants.take_labels +
            (['Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices'] if return_all_labels else [])]

def preprocess_CXP(split, only_frontal, return_all_labels = False):
    details = pd.read_csv(Constants.CXP_details)[['PATIENT', 'PRIMARY_RACE']]
    details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

    split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
    split['Age'] = np.where(split['Age']>=80, 81, split['Age'])

    copy_subjectid = split['subject_id']
    split = split.drop(columns = ['subject_id']).replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81],
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])

    split['subject_id'] = copy_subjectid.astype(str)
    split['Sex'] = np.where(split['Sex']=='Female', 'F', split['Sex'])
    split['Sex'] = np.where(split['Sex']=='Male', 'M', split['Sex'])
    split = split.rename(
        columns = {
            'Pleural Effusion':'Effusion',
            'Lung Opacity': 'Airspace Opacity'
        })
    split['path'] = Constants.image_paths['CXP'] + split['Path'].astype(str)
    split['frontal'] = (split['Frontal/Lateral'] == 'Frontal')
    if only_frontal:
        split = split[split['frontal']]
    split['env'] = 'CXP'
    split['study_id'] = split['path'].apply(lambda x: x[x.index('patient'):x.rindex('/')])

    split = pd.merge(split, details, on = 'subject_id', how = 'inner')
    split = split[(~split.PRIMARY_RACE.isin(['Unknown', 'Patient Refused'])) & (~pd.isnull(split.PRIMARY_RACE))]

    def cat_race(r):
        if r.startswith('White'):
            return 0
        elif r.startswith('Black'):
            return 1
        elif 'Hispanic' in r and 'non-Hispanic' not in r:
            return 2
        elif 'Asian' in r:
            return 3
        else:
            return 4

    split['race'] = split['PRIMARY_RACE'].apply(cat_race)
    split['Sex'] = (split['Sex'] == 'M').astype(int)

    return split[['subject_id','path','Sex',"Age", 'env', 'frontal','study_id', 'race'] + Constants.take_labels +
                (['Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices'] if return_all_labels else [])]

def get_process_func(env):
    if env == 'MIMIC':
        return preprocess_MIMIC
    elif env == 'CXP':
        return preprocess_CXP
    else:
        raise NotImplementedError
