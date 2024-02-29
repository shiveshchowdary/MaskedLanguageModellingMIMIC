MAX_LEN = 256
batch_size = 32
d_model = 128
num_heads = 8
N = 6
num_variables = 18 
num_variables += 1 #for no variable embedding while doing padding
d_ff = 512
epochs = 75
learning_rate = 1e-5
drop_out = 0.1
sinusoidal = False
th_val_roc = 0.84
th_val_pr = 0.48
Uniform = True
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting',True)

import torch
import torch.nn as nn
from utils import MaskedMimicDataSetInHospitalMortality
from torch.utils.data import Dataset, DataLoader, random_split
from model import Model
from torch.nn import functional as F

from tqdm import tqdm
from normalizer import Normalizer
from categorizer import Categorizer

train_data_path_inhospital = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train_listfile.csv"
val_data_path_inhospital = "/data/datasets/mimic3_18var/root/in-hospital-mortality/val_listfile.csv"

train_data_path_phenotyping = "/data/datasets/mimic3_18var/root/phenotyping/train_listfile.csv"
val_data_path_phenotyping = "/data/datasets/mimic3_18var/root/phenotyping/val_listfile.csv"

train_data_path_decompensation = "/data/datasets/mimic3_18var/root/decompensation/train_listfile.csv"
val_data_path_decompensation = "/data/datasets/mimic3_18var/root/decompensation/val_listfile.csv"

data_dir_inhospital = "/data/datasets/mimic3_18var/root/in-hospital-mortality/train/"
data_dir_phenotyping = "/data/datasets/mimic3_18var/root/phenotyping/train/"
data_dir_decompensation = "/data/datasets/mimic3_18var/root/decompensation/train/"

import pickle

with open('normalizer.pkl', 'rb') as file:
    normalizer = pickle.load(file)

with open('categorizer.pkl', 'rb') as file:
    categorizer = pickle.load(file)
    

mean_variance = normalizer.mean_var_dict
cat_dict = categorizer.category_dict


train_ds_inhospital = MaskedMimicDataSetInHospitalMortality(data_dir_inhospital, train_data_path_inhospital, mean_variance, cat_dict, 'training', MAX_LEN)
val_ds_inhospital = MaskedMimicDataSetInHospitalMortality(data_dir_inhospital, val_data_path_inhospital, mean_variance, cat_dict, 'validation', MAX_LEN)

train_ds_phenotyping = MaskedMimicDataSetInHospitalMortality(data_dir_phenotyping, train_data_path_phenotyping, mean_variance, cat_dict, 'training', MAX_LEN)
val_ds_phenotyping = MaskedMimicDataSetInHospitalMortality(data_dir_phenotyping, val_data_path_phenotyping, mean_variance, cat_dict, 'validation', MAX_LEN)

train_ds_decompensation = MaskedMimicDataSetInHospitalMortality(data_dir_decompensation, train_data_path_decompensation, mean_variance, cat_dict, 'training', MAX_LEN)
val_ds_decompensation = MaskedMimicDataSetInHospitalMortality(data_dir_decompensation, val_data_path_decompensation, mean_variance, cat_dict, 'validation', MAX_LEN)


train_dataloader_inhospital = DataLoader(train_ds_inhospital, batch_size = batch_size, shuffle=True)
val_dataloader_inhospital = DataLoader(val_ds_inhospital, batch_size = 1)

train_dataloader_phenotyping = DataLoader(train_ds_phenotyping, batch_size = batch_size, shuffle=True)
val_dataloader_phenotyping = DataLoader(val_ds_phenotyping, batch_size = 1)

train_dataloader_decompensation = DataLoader(train_ds_decompensation, batch_size = batch_size, shuffle=True)
val_dataloader_decompensation = DataLoader(val_ds_decompensation, batch_size = 1)

model = Model(d_model, num_heads, d_ff, num_variables, N, sinusoidal).to(DEVICE)
criterion = nn.MSELoss(reduce=False)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

best_val_loss = float('inf')
early_stopping_counter = 0
patience = 10 

def calculate_loss(model, data_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs['encoder_input'], inputs['encoder_mask'])
            outputs = torch.where(torch.logical_or(pretraining_mask==0,pretraining_mask==-1), torch.tensor(0.0), outputs)
            labels = torch.where(torch.logical_or(pretraining_mask==0,pretraining_mask==-1), torch.tensor(0.0), batch['labels'])
            loss = criterion(outputs, labels)
            loss = torch.where(pretraining_mask==1, loss, torch.zeros_like(loss))
            loss = torch.sum(loss)
            N = torch.sum(pretraining_mask == 1).item()
            loss /= N
            total_loss += loss.item()
    return total_loss/len(data_loader)

for epoch in range(1):
    total_loss = 0
    model.train()
    n = 0
    for batch in tqdm(train_dataloader_decompensation, desc=f'Epoch {epoch + 1}/{epochs}', leave=False, mininterval=1):
        inp = batch['encoder_input']
        mask = batch['encoder_mask']
        pretraining_mask = batch['pretraining_mask']
        outputs = model(inp, mask)
        outputs = torch.where(torch.logical_or(pretraining_mask==0,pretraining_mask==-1), torch.tensor(0.0), outputs)
        labels = torch.where(torch.logical_or(pretraining_mask==0,pretraining_mask==-1), torch.tensor(0.0), batch['labels'])
        loss = criterion(outputs, labels)
        loss = torch.where(pretraining_mask==1, loss, torch.zeros_like(loss))
        loss = torch.sum(loss)
        N = torch.sum(pretraining_mask == 1).item()
        loss /= N
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n+=1
        if n%500 == 0:
            val_loss = calculate_loss(model, val_dataloader_inhospital)
            print(f'Epoch {epoch + 1}/{epochs} batches {n}, Validation Loss: {val_loss:.3f}', end='\r')
    val_loss = calculate_loss(model, val_dataloader_inhospital)
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss/len(train_dataloader_inhospital):.3f}')
    print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.3f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break