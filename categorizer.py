import pickle 
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
pd.set_option('future.no_silent_downcasting',True)


category_config = {
    "Glascow coma scale verbal response": {
            "No Response-ETT": 1,
            "No Response": 1,
            "1 No Response": 1,
            "1.0 ET/Trach": 1,
            "2 Incomp sounds": 2,
            "Incomprehensible sounds": 2,
            "3 Inapprop words": 3,
            "Inappropriate Words": 3,
            "4 Confused": 4,
            "Confused": 4,
            "5 Oriented": 5,
            "Oriented": 5
    },
    "Glascow coma scale eye opening": {
            "None": 0,
            "1 No Response": 1,
            "2 To pain": 2, 
            "To Pain": 2,
            "3 To speech": 3, 
            "To Speech": 3,
            "4 Spontaneously": 4,
            "Spontaneously": 4
        },
    "Glascow coma scale motor response": {
            "1 No Response": 1,
            "No response": 1,
            "2 Abnorm extensn": 2,
            "Abnormal extension": 2,
            "3 Abnorm flexion": 3,
            "Abnormal Flexion": 3,
            "4 Flex-withdraws": 4,
            "Flex-withdraws": 4,
            "5 Localizes Pain": 5,
            "Localizes Pain": 5,
            "6 Obeys Commands": 6,
            "Obeys Commands": 6
        }
}

class Categorizer:
    def __init__(self, data, data_dir):
        self.category_dict = category_config
        self.data = data
        self.data_dir = data_dir
        
train_data_path = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train_listfile.csv"
val_data_path = "/data/datasets/mimic3_18var/root/in-hospital-mortality/val_listfile.csv"

data_dir = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train/"


save = False
if save:
    categorizer = Categorizer(pd.read_csv(train_data_path), data_dir)
    with open("categorizer.pkl", "wb") as file:
        pickle.dump(categorizer, file)

    print("Completed Saving Categorizer........")