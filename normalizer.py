import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm
import pickle
pd.set_option('future.no_silent_downcasting',True)

class Normalizer:
    def __init__(self, data, data_dir):
        self.data = data
        self.data_dir = data_dir
        self.categorical_variables = ['Glascow coma scale eye opening', 
                                 'Glascow coma scale motor response', 
                                 'Glascow coma scale verbal response']
        self.mean_var_dict = self.get_mean_var()
        
        
    def get_mean_var(self):
        sample_path = self.data_dir + self.data['stay'][0]
        id_name_dict = {}
        df = pd.read_csv(sample_path)
        df.drop(labels=self.categorical_variables, axis=1, inplace=True)
        for i in range(len(df.columns)):
            id_name_dict[i] = df.columns[i]
        variable_values = {k : [] for k in df.columns[1:]}
        for sample_path in tqdm(self.data['stay']):
            sample_path = self.data_dir+sample_path
            df = pd.read_csv(sample_path)
            values = df.values
            df.drop(labels=self.categorical_variables, axis=1, inplace=True)
            df.replace(['ERROR','no data','.','-','/','VERIFIED'], np.nan, inplace=True)
            cols = df.columns[1:]
            df = df[cols]
            values = df.values
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if self.isNAN(values[i][j]) == False:
                        variable_values[id_name_dict[j+1]].append(float(values[i][j]))
        result_dict = {}
        for feature, values in variable_values.items():
            mean_value = np.mean(values)
            variance_value = np.var(values)
            result_dict[feature] = {'mean': mean_value, 'variance': variance_value}
        return result_dict
    def isNAN(self, val):
        return val!=val
    

train_data_path = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train_listfile.csv"
val_data_path = "/data/datasets/mimic3_18var/root/in-hospital-mortality/val_listfile.csv"

data_dir = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train/"


save = False
if save:
    normalizer = Normalizer(pd.read_csv(train_data_path), data_dir)
    with open('normalizer.pkl', 'wb') as file:
        pickle.dump(normalizer, file)

    print("Completed Saving Normalizer........")