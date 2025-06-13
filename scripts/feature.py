import torch
from torch_geometric.data import Data
import pickle
import numpy as np
import os
from collections import OrderedDict
# from torchdrug import data
from torchdrug import layers
from torchdrug.layers import geometry
import pickle
import torchdrug
from torchdrug import core, models, tasks
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
# with open('/mnt/storage1/guanmm/esm/esm.pkl', 'rb') as f:
with open('/mnt/storage1/guanmm/New/customed_data/PP/feature/pre_pp.pkl', 'rb') as f:
    prefea_dict = pickle.load(f)

RSA_dict = pickle.load(open('/mnt/storage1/guanmm/New/customed_data/PP/feature/RSA_dict.pkl', 'rb'))

with open('/mnt/storage1/guanmm/New/customed_data/PPprojection_pp14.pkl', 'rb') as f:
    projections = pickle.load(f)
# subset = list(projections.items())

# 下面这三换换，重新打包特征
train_data = pickle.load(open('/mnt/storage1/guanmm/New/name/pp_name_Train_new.pkl', 'rb'))  # 纯名字
test_data = pickle.load(open('/mnt/storage1/guanmm/New/name/pp_name_Test_new.pkl', 'rb'))

with open('/mnt/storage1/guanmm/New/name/pp_name_label_new.pkl', 'rb') as f:     # 蛋白质名称和标签字典
    all_data_label = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

threshold = 13

def feature_Adj(data_list, label, mode):
    Datasets = []
    for protein in data_list:
        projection = projections[protein]
        RSA = []
        for k, project in zip(range(len(label[protein])), projection):
            RSA.append(RSA_dict[protein][k][0])
        pos = np.where(np.array(RSA) >= 0.00)[0].tolist()
        pos1 = np.where(np.array(RSA) < 0.05)[0].tolist()

        all_neighbors = [torch.tensor(p['neigh_index']) for p in projection]  # List[N, K_i]
        all_dij = [torch.tensor(p['dist']) for p in projection]               # List[N, K_i, 3]
        all_proj = [torch.tensor(p['projections']).reshape(-1, 3) for p in projection]  # [N, K_i, 3]

        numeric_labels = [float(item) for item in label[protein]]
        labels = torch.tensor(numeric_labels, dtype=torch.float)

        pad_val = 0
        xyz_id_tensor = pad_sequence(all_neighbors, batch_first=True, padding_value=pad_val).to(device)   # [N, K]
        p_ij_tensor = pad_sequence(all_proj, batch_first=True, padding_value=0.0).to(device)              # [N, K, 3]

        window_list = []
        valid_mask = []
        for dij in all_dij:
            dist = torch.sqrt((dij ** 2).sum(-1))
            mask = dist < threshold
            win = torch.exp(-(dist ** 2) / (2 * threshold ** 2))
            window_list.append(win)
            valid_mask.append(mask)

        # padding window + mask
        window_padded = pad_sequence(window_list, batch_first=True, padding_value=0.0).unsqueeze(1).to(device)  # [N, 1, K]
        mask_padded = pad_sequence(valid_mask, batch_first=True, padding_value=0).to(device)                   # [N, K]

        prefea = torch.tensor(prefea_dict.get(protein), dtype=torch.float).to(device)

        data = Data(y=labels)
        data.name = protein
        data.length = len(labels)
        data.prefea = prefea
        data.xyz_id_tensor = xyz_id_tensor
        data.p_ij_tensor = p_ij_tensor
        data.window_ij_t = window_padded
        data.valid_mask = mask_padded
        data.POS = pos
        data.POS1 = pos1
        Datasets.append(data)

    with open(f'/mnt/storage1/guanmm/New/customed_data/PP/feature/{mode}_feature_train_dist0.pkl', 'wb') as f:
        pickle.dump(Datasets, f)

feature_Adj(test_data, all_data_label, 'test')


feature_Adj(train_data, all_data_label, 'train')
