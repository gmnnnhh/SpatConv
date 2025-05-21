import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import argparse
from tqdm import tqdm
import Bio.PDB
from model_dist_zn import Spatom

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def def_atom_features():
    A = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 3, 0]}
    V = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 3, 0],
         'CG2': [0, 3, 0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 1, 1]}
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 1], 'CG': [0, 2, 1],
         'CD': [0, 2, 1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 1, 0],
         'CD1': [0, 3, 0], 'CD2': [0, 3, 0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 2, 0],
         'CG2': [0, 3, 0], 'CD1': [0, 3, 0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 2, 0], 'CD': [0, 2, 0], 'NE': [0, 1, 0], 'CZ': [1, 0, 0], 'NH1': [0, 2, 0], 'NH2': [0, 2, 0]}
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [-1, 0, 0],
         'OD1': [-1, 0, 0], 'OD2': [-1, 0, 0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [-1, 0, 0], 'OE1': [-1, 0, 0], 'OE2': [-1, 0, 0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'OG': [0, 1, 0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'OG1': [0, 1, 0],
         'CG2': [0, 3, 0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'SG': [-1, 1, 0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 0, 0],
         'OD1': [0, 0, 0], 'ND2': [0, 2, 0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 0, 0], 'OE1': [0, 0, 0], 'NE2': [0, 2, 0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'ND1': [-1, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'NE2': [-1, 1, 1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 2, 0], 'CE': [0, 2, 0], 'NZ': [0, 3, 1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 0, 1],
         'OH': [-1, 1, 0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'SD': [0, 0, 0], 'CE': [0, 3, 0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 0, 1], 'NE1': [0, 1, 1], 'CE2': [0, 0, 1], 'CE3': [0, 1, 1],
         'CZ2': [0, 1, 1], 'CZ3': [0, 1, 1], 'CH2': [0, 1, 1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                     'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    return atom_features
def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C',
                'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
                'LYS': 'K', 'ARG': 'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path, 'r')
    pdb_res = pd.DataFrame(columns=['ID', 'atom', 'res', 'res_id', 'xyz', 'B_factor'])
    res_id_list = []
    try_list = []
    counter = 0
    before_res_pdb_id = None
    Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19, 'CO': 59,
                            'V': 51,
                            'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3, 'NA': 23, 'HG': 200.6,
                            'MN': 55,
                            'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9, 'SE': 79, 'NI': 58.7}
    encountered_ter = False  # 用于记录是否遇到过 'TER' 行
    with open(file_path, 'r') as pdb_file:
        while True:
            line = pdb_file.readline()
            # 如果行以'ATOM'开头，处理该行
            if line.startswith('ATOM'):
                # 获取原子类型
                atom_type = line[76:78].strip()
                if atom_type not in Relative_atomic_mass.keys():
                    continue
                atom_count += 1
                # 获取残基ID，并在需要时增加残基计数
                res_pdb_id = int(line[22:26])
                if res_pdb_id != before_res_pdb_id:
                    res_count += 1
                before_res_pdb_id = res_pdb_id
                if line[12:16].strip() not in ['N', 'CA', 'C', 'O', 'H']:
                    is_sidechain = 1
                else:
                    is_sidechain = 0
                res = res_dict[line[17:20]]
                atom = line[12:16].strip()
                try:
                    atom_fea = atom_fea_dict[res][atom]
                except KeyError:
                    atom_fea = [0.5, 0.5, 0.5]
                raw_value = line[22:26]
                last_col = line[26:27]  # 最后一列
                last_col_value = 0
                if last_col.isalpha():
                    last_col_value = ord(last_col) - ord('A') + 1
                combined_value = int(''.join(raw_value)) * 100 + last_col_value
                if len(try_list) == 0:
                    try_list.append(combined_value)
                    counter += 1
                elif try_list[-1] != int(combined_value):
                    try_list.append(combined_value)
                    counter += 1

                tmps = pd.Series(
                    {'ID': atom_count, 'atom': line[12:16].strip(), 'atom_type': atom_type, 'res': res,
                     'res_id': counter,
                     'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                     'mass': Relative_atomic_mass[atom_type], 'is_sidechain': is_sidechain})

                if len(res_id_list) == 0:
                    res_id_list.append(combined_value)
                elif res_id_list[-1] != int(combined_value):
                    res_id_list.append(combined_value)

                pdb_res = pdb_res._append(tmps, ignore_index=True)

            elif line.startswith('TER'):
                encountered_ter = True

            elif not line:
                res_id_list = list(range(1, len(res_id_list) + 1))   # 反复修改哈 序号有重复的，重新编号！
                break

            if encountered_ter and line.strip():
                encountered_ter = False

        return pdb_res, res_id_list


def extract_features(pdb_file_path):
    pdb_res, res_id_list = get_pdb_DF(pdb_file_path)
    features = []
    for index, row in pdb_res.iterrows():
        xyz = row['xyz']
        mass = row['mass']
        is_sidechain = row['is_sidechain']
        features.append([xyz[0], xyz[1], xyz[2], mass, is_sidechain])
    return np.array(features)

def calculate_projections(seq, residue_psepos_CA, residue_psepos_SC, dist=12, max_neighbors=64):
    projections = []
    pos_CA = residue_psepos_CA[seq]
    pos_SC = residue_psepos_SC[seq]
    pr_center = np.mean(pos_CA, axis=0)
    epsilon = torch.tensor(1e-6, dtype=torch.float32)

    for i in range(len(pos_CA)):
        res_centers = pos_CA[i]
        res_zaxis = pos_SC[i]
        res_yaxis = pos_SC[i - 1]

        delta_10 = torch.from_numpy(res_zaxis - res_centers).float()
        delta_20 = torch.from_numpy(res_yaxis - res_centers).float()

        zaxis = (delta_10 + epsilon) / (torch.sqrt(torch.sum(delta_10 ** 2, dim=-1, keepdim=True)) + epsilon)
        yaxis = torch.cross(zaxis, delta_20)
        yaxis = (yaxis + epsilon) / (torch.sqrt(torch.sum(yaxis ** 2, dim=-1, keepdim=True)) + epsilon)
        xaxis = torch.cross(yaxis, zaxis)
        xaxis = (xaxis + epsilon) / (torch.sqrt(torch.sum(xaxis ** 2, dim=-1, keepdim=True)) + epsilon)

        res_dist = np.sqrt(np.sum((pos_CA - res_centers) ** 2, axis=1))
        neigh_index = np.where(res_dist < dist)[0]
        res_pos = pos_CA[neigh_index] - res_centers

        translation = torch.tensor(res_pos).float()
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

    return projections

def get_sequence_from_pdb(pdb_file_path):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if Bio.PDB.is_aa(residue):
                    sequence += Bio.PDB.Polypeptide.three_to_one(residue.resname)
    return sequence

def get_pdb_DF(pdb_file_path):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    atoms = []
    res_id_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if Bio.PDB.is_aa(residue):
                    res_id = residue.id[1]
                    res_name = Bio.PDB.Polypeptide.three_to_one(residue.resname)
                    for atom in residue:
                        atom_name = atom.name
                        atom_coord = atom.coord
                        atom_mass = atom.mass
                        is_sidechain = 1 if atom_name not in ['N', 'CA', 'C', 'O'] else 0
                        atoms.append([res_id, atom_name, res_name, atom_coord, atom_mass, is_sidechain])
                    if res_id not in res_id_list:
                        res_id_list.append(res_id)
    pdb_res = pd.DataFrame(atoms, columns=['res_id', 'atom', 'res', 'xyz', 'mass', 'is_sidechain'])
    return pdb_res, res_id_list

def cal_Psepos(seqlist, pdb_file_path, psepos):
    seq_CA_pos = {}
    seq_sidechain_centroid = {}

    for seq_id in tqdm(seqlist):
        pdb_res_i, res_id_list = get_pdb_DF(pdb_file_path)

        res_CA_pos = []
        res_sidechain_centroid = []
        res_types = []

        for res_id in res_id_list:
            res_type = pdb_res_i[pdb_res_i['res_id'] == res_id]['res'].values[0]
            res_types.append(res_type)

            res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
            xyz = np.array(res_atom_df['xyz'].tolist())
            masses = np.array(res_atom_df['mass'].tolist()).reshape(-1, 1)
            centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_atom_df = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['is_sidechain'] == 1)]

            try:
                CA = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['atom'] == 'CA')]['xyz'].values[0]
            except IndexError:
                print('IndexError: no CA in seq:{} res_id:{}'.format(seq_id, res_id))
                CA = centroid

            res_CA_pos.append(CA)

            if len(res_sidechain_atom_df) == 0:
                res_sidechain_centroid.append(centroid)
            else:
                xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
                masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
                sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
                res_sidechain_centroid.append(sidechain_centroid)

        res_CA_pos = np.array(res_CA_pos)
        res_sidechain_centroid = np.array(res_sidechain_centroid)

        seq_CA_pos[seq_id] = res_CA_pos
        seq_sidechain_centroid[seq_id] = res_sidechain_centroid

    if psepos == 'CA':
        return seq_CA_pos
    elif psepos == 'SC':
        return seq_sidechain_centroid

def feature_Adj(protein, projections, prefea, label):
    projection = projections  # 每个蛋白质所有邻居投影

    all_neighbours_dict = {}
    all_neighbours_project = {}
    all_neighbours_dist = {}

    for k, project in zip(range(len(label)), projection):
        index_list = project['neigh_index']
        index_list = torch.tensor(index_list)
        all_neighbours_dict[k] = index_list

        dij = project['dist']
        dij = torch.tensor(dij)
        all_neighbours_dist[k] = dij

        projection_neigh = project['projections']
        tensor_matrix = torch.tensor(projection_neigh).float()
        all_neighbours_project[k] = tensor_matrix

    dis_dij = all_neighbours_dist
    xyz_id = all_neighbours_dict

    data = Data(x=prefea)
    data.name = protein
    data.length = len(label)
    data.xyz_nb = all_neighbours_project
    data.xyz_id = xyz_id
    data.dij = dis_dij

    return data


def predict(data):
    model = Spatom().to(DEVICE)
    model.load_state_dict(torch.load('/mnt/data0/uploaduser/best_model.dat'))
    model.eval()

    prefea = torch.tensor(data.x).to(DEVICE, dtype=torch.float)
    xyz_nb = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.xyz_nb.items()}
    xyz_id = {key: tensor.to(DEVICE, dtype=torch.long) for key, tensor in data.xyz_id.items()}
    dij = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.dij.items()}

    with torch.no_grad():
        pred = model(prefea, xyz_nb, xyz_id, dij)
        pred = pred.cpu().numpy().tolist()
 # 使用sigmoid函数将预测值转化为0-1之间

    return pred



import csv


def save_results_to_csv(seq_id, sequence, predictions):
    csv_file = f"/mnt/data0/guanmm/{seq_id}_predictions.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Amino Acid", "Prediction Score", "Prediction Class"])

        for i, (aa, pred) in enumerate(zip(sequence, predictions)):
            pred_class = 1 if pred >= 0.21 else 0
            writer.writerow([i + 1, aa, pred, pred_class])

    print(f"Results saved to {csv_file}")


def load_h5_features(feature_path):
    protein_data_dict = {}

    # 打开HDF5文件
    with h5py.File(feature_path, "r") as h5fi:
        for dataset_name in h5fi.keys():
            # 从HDF5文件中加载数据集
            data = h5fi[dataset_name][:]
            protein_data_dict[dataset_name] = data

    return protein_data_dict



def main(args):
    pdb_path = args.pdb_file
    feature_path = args.feature_file

    # 提取PDB文件名作为蛋白质名称和数据集名称
    seq_id = os.path.splitext(os.path.basename(pdb_path))[0]

    # 提取序列和特征
    sequence = get_sequence_from_pdb(pdb_path)
    # print(len(sequence))

    # pdb_features = extract_features(pdb_path)

    train_list = [seq_id]
    # print(train_list)
    residue_psepos_CA = cal_Psepos(train_list, pdb_path, 'CA')

    # array_data = residue_psepos_CA['5dvz_AB']  就这算少了呢
    # print("Number of rows:", array_data.shape[0])
    residue_psepos_SC = cal_Psepos(train_list, pdb_path, 'SC')

    projections = calculate_projections(seq_id, residue_psepos_CA, residue_psepos_SC)

    try:
        # 直接通过PDB文件名加载HDF5特征
        protein_data_dict = load_h5_features(feature_path)
        prefea = protein_data_dict[seq_id]
    except Exception as e:
        print(f"Error loading feature file: {e}")
        return

    label = [0] * len(prefea)  # 伪标签
    data = feature_Adj(seq_id, projections, prefea, label)

    predictions = predict(data)

    # 生成并保存CSV文件
    save_results_to_csv(seq_id, sequence, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict protein features from PDB and HDF5 files")

    parser.add_argument('--pdb_file', '-p', required=True, help='Path to the PDB file')
    parser.add_argument('--feature_file', '-f', required=True, help='Path to the HDF5 feature file')

    args = parser.parse_args()
    main(args)