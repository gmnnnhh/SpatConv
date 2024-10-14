import os
import pickle
import torch
from tqdm import tqdm
import numpy as np

dist = 13

Dataset_dir = '/mnt/storage1/guanmm/New/customed_data/PP'

with open(Dataset_dir + '/pp_psepos_CA.pkl', 'rb') as f:
    residue_psepos_CA = pickle.load(f)

with open(Dataset_dir + '/pp_psepos_SC.pkl', 'rb') as f:
    residue_psepos_SC = pickle.load(f)

    train_list = []
    seqanno = {}
    trainset_anno = Dataset_dir + '/{}'.format('PP_1251.txt')

    with open(trainset_anno, 'r') as f:
        train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        query_seq = train_text[i + 1].strip()
        query_anno = train_text[i + 2].strip()
        train_list.append(query_id)
        seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}

    all_local_coordinate_frames = []

    for s, (dataset, seqlist) in enumerate(zip(['train'], [train_list])):
        data_dict = {}
        for seq in tqdm(seqlist):
            PDB_DF_dir = Dataset_dir + '/PDB_DF'
            with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq), 'rb') as f:
                tmp = pickle.load(f)
            pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
            res_types = []
            for res_id in res_id_list:
                res_type = pdb_res_i[pdb_res_i['res_id'] == res_id]['res'].values[0]
                res_types.append(res_type)
            label = ''.join(res_types)
            seq_data = []
            pos_CA = residue_psepos_CA[seq]
            pos_SC = residue_psepos_SC[seq]
            pr_center = np.mean(pos_CA, axis=0)
            epsilon = torch.tensor(1e-6)
            projections = []
            for i in range(len(label)):
                res_centers = pos_CA[i]
                res_zaxis = pos_SC[i]
                if i == 0:
                    res_yaxis = (1, 0, 0)
                else:
                    res_yaxis = pos_SC[i - 1]
                delta_10 = torch.from_numpy(res_zaxis - res_centers, )
                delta_20 = torch.from_numpy(res_yaxis - res_centers)
                zaxis = (delta_10 + epsilon) / (
                        torch.sqrt(torch.sum(delta_10 ** 2, dim=-1, keepdim=True)) + epsilon)
                yaxis = torch.cross(zaxis, delta_20)
                yaxis = (yaxis + epsilon) / (
                        torch.sqrt(torch.sum(yaxis ** 2, dim=-1, keepdim=True)) + epsilon)
                xaxis = torch.cross(yaxis, zaxis)
                xaxis = (xaxis + epsilon) / (
                        torch.sqrt(torch.sum(xaxis ** 2, dim=-1, keepdim=True)) + epsilon)
                res_dist = np.sqrt(np.sum((pos_CA - res_centers) ** 2, axis=1))
                neigh_index = np.where(res_dist < dist)[0]
                res_pos = pos_CA[neigh_index] - res_centers
                translation = torch.tensor(res_pos)
                projection_list = []
                for row in translation:
                    x = torch.dot(row, xaxis)
                    y = torch.dot(row, yaxis)
                    z = torch.dot(row, zaxis)
                    projection = (x, y, z)
                    projection = np.array(projection)
                    projection = np.round(projection, 2)
                    projection_list.append(projection)
                projection_dict = {
                    'neigh_index': neigh_index,
                    'projections': projection_list,
                    'dist': np.round(res_pos, 3)
                }
                projections.append(projection_dict)
            data_dict[seq] = projections
        f = open('/mnt/storage1/guanmm/New/customed_data/PP' + 'projection.pkl', 'wb')
        pickle.dump(data_dict, f)
print('done')
