import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
pd.set_option('future.no_silent_downcasting',True)


class MaskedMimicDataSetInHospitalMortality(Dataset):
    def __init__(self, data_dir, csv_file, mean_variance , cat_dict, mode, seq_len, pad_value = 0, device = DEVICE):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.mode = mode
        self.data_df = pd.read_csv(csv_file)
        self.mean_variance = mean_variance
        self.pad_value = pad_value
        self.device = device
        self.cat_dict = cat_dict
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        path = self.data_dir + self.data_df['stay'][idx]
        data = pd.read_csv(path)
        
        id_name_dict = {}
        data.replace(['ERROR','no data','.','-','/','VERIFIED','CLOTTED',"*",'ERROR DISREGARD PREVIOUS RESULT OF 32','DISREGARD PREVIOUSLY REPORTED 33'], np.nan, inplace=True)
        for i in range(len(data.columns)):
            id_name_dict[i] = data.columns[i]
        values = data.values
        sample = self.uniform_extract(values, id_name_dict, 25)
        if len(sample[0]) >= self.seq_len :
            sample[0] = sample[0][:self.seq_len]
            sample[1] = sample[1][:self.seq_len]
            sample[2] = sample[2][:self.seq_len]
            sample[3] = sample[3][:self.seq_len]
        num_padd_tokens = self.seq_len - len(sample[0])
        sample[0].reverse()
        sample[1].reverse()
        sample[2].reverse()
        sample[3].reverse()
        mask_len = len(sample[0])
        mask = torch.rand(mask_len) < 0.1 # randomly masking 10 percent of the values
        val_input = torch.tensor(sample[1], dtype=torch.float)
        val_input[mask] = 0
        variable_input = torch.cat([
            torch.tensor(sample[2], dtype=torch.int64),
            torch.tensor([self.pad_value]*num_padd_tokens, dtype=torch.int64)
        ])
        value_input = torch.cat([
            val_input,
            torch.tensor([self.pad_value]*num_padd_tokens, dtype=torch.float)
        ])
        val = torch.tensor(sample[0], dtype=torch.float)
        time_input = torch.cat([
             val - val.min() ,
            torch.tensor([self.pad_value]*num_padd_tokens, dtype=torch.float)
        ])
        variables = sample[3] + ['pad token']*num_padd_tokens
        
        assert variable_input.size(0) == self.seq_len
        assert value_input.size(0) == self.seq_len
        assert time_input.size(0) == self.seq_len
        
        return {
            "encoder_input" : [time_input.to(self.device), variable_input.to(self.device), value_input.to(self.device)],
            "encoder_mask": (variable_input != self.pad_value).unsqueeze(0).int().to(self.device),
            "pretraining_mask": torch.cat([mask, torch.tensor([-1]*num_padd_tokens)]).int().to(self.device),# filling pad tokens with -1 in pretraining mask
            "labels":torch.cat([
                torch.tensor(sample[1], dtype=torch.float),
                torch.tensor([self.pad_value]*num_padd_tokens, dtype=torch.float)
            ]).to(DEVICE),
            "variables" : variables,
        }
    
    def extract(self, values, id_name_dict):
        sample = [[],[],[],[]]
        for i in range(values.shape[0]):
            i = values.shape[0] - 1 - i
            time = values[i,0]
            for j in range(1, values.shape[1]):
                if self.isNAN(values[i][j]) == False:
                    if id_name_dict[j] in self.cat_dict.keys():
                        sample[0].append(time)
                        sample[1].append(self.cat_dict[id_name_dict[j]][values[i][j]])
                        sample[2].append(j)
                        sample[3].append(id_name_dict[j])
                    else:
                        mean = self.mean_variance[id_name_dict[j]]['mean']
                        var = self.mean_variance[id_name_dict[j]]['variance']
                        val = (float(values[i][j]) - mean)/var
                        sample[0].append(time)
                        sample[1].append(val)
                        sample[2].append(j)
                        sample[3].append(id_name_dict[j])
        return sample
    def uniform_extract(self, values, id_name_dict, threshold):
        sample = [[],[],[],[]]
        count_dict = {}
        for i in range(values.shape[0]):
            i = values.shape[0] - 1 - i
            time = values[i,0]
            for j in range(1, values.shape[1]):
                if self.isNAN(values[i][j]) == False:
                    if id_name_dict[j] in self.cat_dict.keys() :
                        if id_name_dict[j] not in count_dict:
                            count_dict[id_name_dict[j]] = 0
                        if count_dict[id_name_dict[j]] <= threshold:
                            sample[0].append(time)
                            sample[1].append(self.cat_dict[id_name_dict[j]][values[i][j]])
                            sample[2].append(j)
                            sample[3].append(id_name_dict[j])
                            count_dict[id_name_dict[j]]+=1
                    else:
                        if id_name_dict[j] not in count_dict:
                            count_dict[id_name_dict[j]] = 0
                        if count_dict[id_name_dict[j]] <= threshold:
                            mean = self.mean_variance[id_name_dict[j]]['mean']
                            var = self.mean_variance[id_name_dict[j]]['variance']
                            val = (float(values[i][j]) - mean)/var
                            sample[0].append(time)
                            sample[1].append(val)
                            sample[2].append(j)
                            sample[3].append(id_name_dict[j])
                            count_dict[id_name_dict[j]]+=1
        return sample
    def isNAN(self, val):
        return val!=val
