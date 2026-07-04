'''
this file is same as data_generator.py in DLM_generate but it is for triviaqa and llada_instruct.
both script is doinf same job but for different setup.

'''
import torch
import random
import os
from utils.data.preprocess import RWDataset

set_seed = 42
random.seed(set_seed)

dataset_name = "triviaqa"
task_name = "entropy"
model_name = "Instruct"
gen_length = 64

data_path = f"./DLM_generate/process_data/{dataset_name}_{model_name}_{gen_length}_{task_name}/"


def load_split(split):
    X = torch.load(data_path + f'{dataset_name}_{split}.pt')
    if isinstance(X, list):
        X = torch.stack([torch.tensor(x) for x in X]).squeeze()
    X = X.transpose(0, 1)  # [time_length, num_data, feature_dim]

    Y = torch.load(data_path + f'{dataset_name}_{split}_labellist.pt')
    if isinstance(Y, list):
        Y = torch.tensor(Y)

    assert X.shape[1] == len(Y), f"Mismatch: X has {X.shape[1]} samples, Y has {len(Y)}"

    time_step = X.shape[0]
    num_sample = X.shape[1]
    T = torch.arange(1, time_step + 1).unsqueeze(1).repeat(1, num_sample)

    print(f"[{split}] X: {X.shape}, T: {T.shape}, Y: {len(Y)}")
    return X, T, Y


USE_TEST_FOR_ALL = True

if USE_TEST_FOR_ALL:
    X_test, T_test, Y_test = load_split("test")
    X_train, T_train, Y_train = X_test, T_test, Y_test
    X_val, T_val, Y_val = X_test, T_test, Y_test
else:
    X_train, T_train, Y_train = load_split("train")
    X_val, T_val, Y_val = load_split("val")
    X_test, T_test, Y_test = load_split("test")


D = {
    'train_loader': RWDataset(X_train, T_train, Y_train),
    'val': (X_val, T_val, Y_val),
    'test': (X_test, T_test, Y_test),
    'gt_exps': None
}

torch.save(D, data_path + f'{dataset_name}_ent_labellist_split.pt')
torch.save(D, data_path + 'split=1.pt')

print(f"\nSaved split file to: {data_path}split=1.pt")