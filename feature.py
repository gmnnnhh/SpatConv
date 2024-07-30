import torch
from torch_geometric.data import Data
import pickle
import numpy as np
import os
from collections import OrderedDict
from torchdrug import layers
from torchdrug.layers import geometry
import pickle
import torchdrug
from torchdrug import core, models, tasks
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import torch.nn as nn

with open('/mnt/storage1/guanmm/New/customed_data/PP/feature/pre_pp.pkl', 'rb') as f:
    prefea_dict = pickle.load(f)

RSA_dict = pickle.load(open('/mnt/storage1/guanmm/New/customed_data/PP/feature/RSA_dict.pkl', 'rb'))

with open('/mnt/storage1/guanmm/New/customed_data/projection.pkl', 'rb') as f:
    projections = pickle.load(f)

Dataset_dir = os.path.abspath('..') + '/Spatom/Data/DBD/data'

train_data = pickle.load(open('/mnt/storage1/guanmm/New/customed_data/PP/pp_name_Train.pkl', 'rb'))

test_data = pickle.load(open('/mnt/storage1/guanmm/New/customed_data/PP/pp_name_Test.pkl', 'rb'))

with open('/mnt/storage1/guanmm/New/customed_data/PP/pp_name_label.pkl', 'rb') as f:
    all_data_label = pickle.load(f)

with open('/mnt/storage1/guanmm/New/customed_data/PP/feature/Dist_dict.pkl', 'rb') as f:
    Dist_dict = pickle.load(f)


def feature_Adj(data_list, label, mode):
    Datasets = []
    for i in data_list:
        # protein = '3R9A_A'
        protein = i
        projection = projections[protein]
        # print(CA)
        # feature = []
        RSA = []
        all_neighbours_dict = {}
        all_neighbours_project = {}
        all_neighbours_dist = {}

        for k, project in zip(range(len(label[i])), projection):
            index_list = project['neigh_index']
            index_list = np.array(index_list)
            index_list = torch.tensor(index_list)
            # print(len(index_list))
            all_neighbours_dict[k] = index_list
            dij = project['dist']
            dij = np.array(dij)
            dij = torch.tensor(dij)
            all_neighbours_dist[k] = dij
            projection_neigh = project['projections']
            matrix = np.array(projection_neigh).reshape(-1, 3)
            tensor_matrix = torch.tensor(matrix)
            all_neighbours_project[k] = tensor_matrix
            AA = []
            AA.extend(RSA_dict[i][k])
            RSA.append(RSA_dict[i][k][0])
        if mode == 'test':
            pos = np.where(np.array(RSA) >= 0.00)[0].tolist()
            pos1 = np.where(np.array(RSA) < 0.05)[0].tolist()
        else:
            pos = np.where(np.array(RSA) >= 0.00)[0].tolist()
            pos1 = np.where(np.array(RSA) < 0.05)[0].tolist()
        if mode == 'test':
            numeric_labels = [float(item) for item in label[protein]]
            labels = torch.tensor(np.array(numeric_labels), dtype=torch.float)
            Dist = edge_weight(torch.tensor(Dist_dict[i]))[pos, :][:, pos]
            adj = torch.tensor(np.where(np.array(Dist_dict[i]) < 14, 1, 0)[pos, :][:, pos])
            xyz_nb = all_neighbours_project
            prefea = prefea_dict.get(i)
            dis_dij = all_neighbours_dist
            xyz_id = all_neighbours_dict
        else:
            numeric_labels = [float(item) for item in label[protein]]
            labels = torch.tensor(np.array(numeric_labels)[pos], dtype=torch.float)
            Dist = edge_weight(torch.tensor(Dist_dict[i]))[pos, :][:, pos]
            adj = torch.tensor(np.where(np.array(Dist_dict[i]) < 14, 1, 0)[pos, :][:, pos])
            xyz_nb = all_neighbours_project
            prefea = prefea_dict.get(i)
            dis_dij = all_neighbours_dist
            xyz_id = all_neighbours_dict

        data = Data(y=labels)
        data.name = i
        data.dist = Dist

        data.POS = pos
        data.POS1 = pos1

        length = len(label[protein])
        data.length = length
        data.xyz_nb = xyz_nb
        data.xyz_id = xyz_id
        data.dij = dis_dij
        data.prefea = prefea
        data.adj = adj
        Datasets.append(data)
    f = open('/mnt/storage1/guanmm/New/customed_data/PP/feature/' + mode + '_feature.pkl', 'wb')
    pickle.dump(Datasets, f)

feature_Adj(test_data, all_data_label, 'test')
feature_Adj(train_data, all_data_label, 'train')
