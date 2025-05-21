import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import numpy as np

# graphbind中0.5

dropout_rate = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ALPHA = 0.7
LAMBDA = 1.5


class GraphChenn(nn.Module):
    '''
    changed from https://github.com/chennnM/GCNII
    '''

    def __init__(self, in_features, out_features):
        super(GraphChenn, self).__init__()
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))  # p 可训练参数矩阵
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0, lamda, alpha, l):
        theta = min(1, math.log(lamda / l + 1))  # β_l 超参数
        # hi = torch.spmm(adj, input)
        hi = input
        # support = (1 - alpha) * hi + alpha * h0 # 去掉初始残差
        support = hi
        r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r  # GCNII
        output = output + input  # 上一层的
        return output


class Spatom(nn.Module):
    '''
    changed from https://github.com/chennnM/GCNII
    '''

    # def __init__(self, nlayers=LAYER, nfeat=63, n_xyz=3, nhidden=HIDDEN, dropout=DROPOUT, lamda=LAMBDA, alpha=ALPHA,
    #              out_dim=1):
    # geobind 里   radius在准地理距离 d_ij 上的高斯窗口的偏差。默认为 1
    def __init__(self, in_channels=1024, out_channels=1, n_layers=3, lamda=LAMBDA,
                 alpha=ALPHA, sigma=14):
        super(Spatom, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ########### GeoBind 开始
        self.Input = in_channels
        self.Output = out_channels

        self.Hidden = 64
        self.Cuts = 16
        self.distance_thresholds = [13]

        self.fc = nn.Linear(in_channels, self.Hidden)

        self.BN = nn.ModuleList(
            [nn.BatchNorm1d(self.Hidden) for iteration in range(n_layers)])
        self.dropout = nn.Dropout(p=0.5)  # p是dropout概率

        # Transformation of the input features:  对输入特征进行变换 xyz_id 这里是邻居特征 预编码
        self.net_in = nn.ModuleList([nn.Sequential(
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),  # 也加入 dropout了 0.5
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2), ) for iteration in range(n_layers)])  #  (H,)
        # self.norm_in = nn.GroupNorm(4, self.Hidden)
        self.net_in0 = nn.Sequential(
            nn.Linear(self.Input, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),  # 也加入 dropout了 0.5
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2), )  #  (H,)

        # 3D convolution filters, encoded as an MLP:  对邻居投影的变换
        self.conv = nn.ModuleList([nn.Sequential(
            nn.Linear(3, self.Cuts),  # (C, 3) + (C,)
            nn.ReLU(),  # KeOps does not support well LeakyReLu
            # n.Dropout(dropout_rate),     # 新加入的dropout 加入麻烦
            nn.Linear(self.Cuts, self.Hidden),
        ) for iteration in range(n_layers)])  # (H, C) + (H,)

        # Transformation of the output features:   后编码
        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Output, self.Output),  # (O, O) + (O,) 降低模型复杂度
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),  # 新加入的dropout  删除这个，效果应该是变好了 存疑
            nn.Sigmoid()  # 添加Sigmoid激活函数 为了多层更新呐
        )
        self.chen = nn.ModuleList(
            [GraphChenn(self.Hidden, self.Hidden) for iteration in range(n_layers)])

        self.alpha = alpha
        self.lamda = lamda
        self.sigma = sigma

        # self.norm_out = nn.GroupNorm(4, self.Output)

        with torch.no_grad():
            for seq_module in self.conv:
                # 初始化第一个线性层
                nn.init.normal_(seq_module[0].weight)
                nn.init.uniform_(seq_module[0].bias)
                seq_module[0].bias *= 0.8 * (seq_module[0].weight ** 2).sum(-1).sqrt()

                # 初始化第三个线性层（在ReLU之后）
                nn.init.uniform_(seq_module[2].weight, a=-1 / np.sqrt(self.Cuts), b=1 / np.sqrt(self.Cuts))
                nn.init.normal_(seq_module[2].bias)
                seq_module[2].bias *= 0.5 * (seq_module[2].weight ** 2).sum(-1).sqrt()

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.Hidden, self.Hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),  # 添加Dropout层   减少一层线形层，降低复杂度
                    nn.Linear(self.Hidden, self.Hidden)
                )
                for iteration in range(n_layers)
            ]
        )

        self.act_fn = nn.ReLU()
        self.fc1 = nn.Linear(self.Hidden * 2, self.Hidden)
        self.fc2 = nn.Linear(self.Hidden * 2, self.Hidden)
        self.fc3 = nn.Linear(self.Hidden * 4, self.Hidden)
        self.mlp = nn.Linear(self.Hidden * 3, self.Hidden)


    def print_hyperparameters_and_architecture(self):
        # 打印超参数
        # print("Hyperparameters:")
        hyperparams = ['Input', 'Output', 'Hidden', 'Cuts', 'alpha', 'lamda', 'sigma', 'distance_thresholds']
        for param in hyperparams:
            print(f'{param} = {getattr(self, param)}')

        # 打印模型架构
        # print("\nModel Architecture:")
        # print(self)

    def extract_features(self, x, xyz_id):
        """
        提取特征的通用函数，根据距离筛选出的邻居信息提取每个氨基酸的特征。
        """
        neighbor_features = {}  # 初始化新的字典来存储邻居特征
        for key, selected_neighbors in xyz_id.items():
            # 这里假设 selected_neighbors 已经是基于距离筛选后的邻居索引列表
            # 使用邻居索引从 x 中提取特征

            neighbor_feats = x[torch.stack(selected_neighbors)]
            # 存储提取的邻居特征
            neighbor_features[key] = neighbor_feats
        return neighbor_features

    # GeoBind 由四个包含准测地线卷积层的块组成。在每个块中，在前编码器和后编码器的准测地线卷积层之前和之后设置了多层感知器（MLP）。
    # 每一层的输出都是其直接输入和经过一系列变换（包括dMaSIFConv几何卷积和线性层处理）的结果的和
    def forward(self, x, xyz_nb, xyz_id, dij, ):
        # print(x.shape)

        x = x.squeeze()

        j = 0
        previous_outputs = []

        for threshold in self.distance_thresholds:
            # 根据距离阈值筛选邻居
            current_xyz_id = {}
            current_xyz_nb = {}
            current_dij = {}

            for key, distances in dij.items():
                euclidean_distances = torch.sqrt(torch.sum(distances ** 2, axis=1))
                valid_indices = torch.arange(len(euclidean_distances))[euclidean_distances < threshold]
                # 更新当前阶段的邻居信息
                current_xyz_id[key] = [xyz_id[key][i] for i in valid_indices]  # 根据距离筛选的邻居索引
                current_xyz_nb[key] = [xyz_nb[key][i] for i in valid_indices]  # 相应的邻居坐标投影
                current_dij[key] = [dij[key][i] for i in valid_indices]  # 相应的距离信息
            neighbor_features = self.extract_features(x, current_xyz_id)

            # 编码邻居特征
            if j == 0:
                residual = self.dropout(self.fc(x))  # 编码残差
                f_j = {key: self.net_in0(value) for key, value in neighbor_features.items()}
            else:
                f_j = {key: self.net_in[j](value) for key, value in neighbor_features.items()}

            # P_ij = {key: self.conv[j](value) for key, value in current_xyz_nb.items()}
            P_ij = {}
            for key, value in current_xyz_nb.items():
                # 检查value是否为列表，如果是，则需要将列表中的Tensor堆叠成一个Tensor
                if isinstance(value, list):
                    value_tensor = torch.stack(value).to(self.device)  # 确保堆叠后的Tensor在正确的设备上
                else:
                    value_tensor = value  # 如果value已经是一个Tensor，直接使用
                # 使用转换后的Tensor调用对应的卷积层
                P_ij[key] = self.conv[j](value_tensor)
                # 不过这个卷积层
                # P_ij[key] = value_tensor
            # d2_ij_dict = {}
            window_ij_dict = {}
            window_ij_t_dict = {}
            for key, value in current_dij.items():
                # 计算距离的平方和
                value_tensor = torch.stack(value)  # 将列表中的Tensor元素堆叠成一个新的Tensor
                d2_ij = (value_tensor ** 2).sum(-1)
                window_ij = torch.exp(-d2_ij / (2 * threshold ** 2))  # 这里也动态调整呢 炼丹的想法一大堆，想要有明确答案的想法
                # d2_ij_dict[key] = d2_ij
                window_ij_dict[key] = window_ij
                window_ij_t_dict[key] = window_ij.unsqueeze(0)

            # 预先分配F_ij_tensor
            F_ij_tensor = torch.zeros(len(P_ij), self.Hidden)
            F_ij_tensor = F_ij_tensor.to(DEVICE)

            # 替换哈达玛的，比哈达玛好一点
            for i, (key, P_value) in enumerate(P_ij.items()):
                f_j_value = f_j[key]

                x_coord = P_value[:, 0].unsqueeze(1)  # [N, 1]
                y_coord = P_value[:, 1].unsqueeze(1)  # [N, 1]
                z_coord = P_value[:, 2].unsqueeze(1)  # [N, 1]
                # 分别乘以特征向量
                x_feature = x_coord * f_j_value
                y_feature = y_coord * f_j_value
                z_feature = z_coord * f_j_value
                # 拼接结果
                concatenated_features = torch.cat((x_feature, y_feature, z_feature), dim=1)  # [N, 3*feature_dim]
                # 通过MLP
                transformed_features = self.mlp(concatenated_features)
                window_ij_t_value = window_ij_t_dict[key]
                F_value = window_ij_t_value @ transformed_features
                F_ij_sum = F_value.sum(dim=0, keepdim=True)
                F_ij_tensor[i] = F_ij_sum

            F_ij_tensor1 = self.BN[j](F_ij_tensor)
            linear_output = self.linear_layers[j](F_ij_tensor1)

            layer_inner = self.act_fn(self.chen[j](linear_output, residual, self.lamda, self.alpha, j + 1))
            # layer_inner = F.elu(self.GAT1(layer_inner, adj, residual, self.lamda, self.alpha, 9))
            x = layer_inner
            j += 1

        output = self.net_out(x)  # 这应该是循环后的最后一步
        return output.squeeze()

# layer_inner = F.elu(self.GAT1(current_features, adj, res, self.lamda, self.alpha, 9))  # 9 一个控制应用多少初始的超参数
# 加入图注意力效果不好，也有提升，但就一层chen目前最好   一层加入这个初始残差，效果是不错，
# 下面是又该加入残差啦，或者拼接上一层，框架复杂起来啦
# print(layer_inner)
