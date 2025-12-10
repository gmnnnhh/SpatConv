import numpy as np
import h5py
import pickle

# 定义一个变量，用于存储需要替换的名称
dataset_base_name = 'MN_train'  # 这里可以动态修改为不同的名字

# 构建文件路径
pre_dict_path = f'/mnt/storage1/guanmm/New/{dataset_base_name}.h5'
dataset_names_file = f'/mnt/storage1/guanmm/New/name/{dataset_base_name}_name.txt'
output_file_path = f'/mnt/storage1/guanmm/New/customed_data/PP/h5_pkl/{dataset_base_name}.pkl'

# 创建一个字典来存储蛋白质数据
protein_data_dict = {}

# 首先，读取包含数据集名称的文本文件
with open(dataset_names_file, 'r') as file:
    dataset_names = [line.strip()[1:] for line in file]

# 现在，打开HDF5文件并读取每个数据集
with h5py.File(pre_dict_path, "r") as h5fi:
    for dataset_name in dataset_names:
        # 访问数据集
        dataset = h5fi[dataset_name]
        # 从数据集读取数据
        data = dataset[:]
        protein_data_dict[dataset_name] = data

# 保存字典到pickle文件
with open(output_file_path, 'wb') as output_file:
    pickle.dump(protein_data_dict, output_file)
