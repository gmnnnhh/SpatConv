import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import numpy as np


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
        hi = input
        support = (1 - alpha) * hi + alpha * h0
        r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r  # GCNII
        output = output + input  # 上一层的
        return output


class SpatConv(nn.Module):
    '''
    changed from https://github.com/chennnM/GCNII
    '''

    def __init__(self, in_channels=1024, out_channels=1, n_layers=1, lamda=LAMBDA,
                 alpha=ALPHA, sigma=14):
        super(SpatConv, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Input = in_channels
        self.Output = out_channels
        self.Hidden = 64
        self.Cuts = 16
        self.fc = nn.Linear(in_channels, self.Hidden)
        self.BN = nn.ModuleList(
            [nn.BatchNorm1d(self.Hidden) for iteration in range(n_layers)])
        self.dropout = nn.Dropout(p=0.5)

        self.net_in = nn.ModuleList([nn.Sequential(
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),  # 也加入 dropout了 0.5
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2), ) for iteration in range(n_layers)])  #  (H,)
        # self.norm_in = nn.GroupNorm(4, self.Hidden)
        self.net_in0 = nn.Sequential(
            nn.Linear(self.Input, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),  # 也加入 dropout了 0.5
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2), )  #  (H,)

        # 3D convolution filters, encoded as an MLP:  对邻居投影的变换
        self.conv = nn.ModuleList([nn.Sequential(
            nn.Linear(3, self.Cuts),  # (C, 3) + (C,)
            nn.ReLU(),  # KeOps does not support well LeakyReLu
            # n.Dropout(dropout_rate),     # 新加入的dropout 加入麻烦
            nn.Linear(self.Cuts, self.Hidden),
        ) for iteration in range(n_layers)])  # (H, C) + (H,)


        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Output, self.Output),  # (O, O) + (O,) 降低模型复杂度
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),
            nn.Sigmoid()
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
                    # nn.Linear(self.Hidden, self.Hidden)
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
        print("Hyperparameters:")
        hyperparams = ['Input', 'Output', 'Hidden', 'Cuts', 'alpha', 'lamda', 'sigma', ]
        for param in hyperparams:
            print(f'{param} = {getattr(self, param)}')

        # 打印模型架构
        print("\nModel Architecture:")
        # print(self)

    def extract_features(self, x, xyz_id_tensor):
        """
        提取每个氨基酸邻居的特征。

        参数:
        x: [B_total, feat_dim]，所有氨基酸的特征（整个 batch 展平后）
        xyz_id_tensor: [B, max_N]，每个氨基酸对应的邻居在 x 中的索引

        返回:
        neighbor_feats: [B, max_N, feat_dim]，每个氨基酸的邻居特征
        """
        # 拉平成 [B * max_N] 的索引向量
        flat_idx = xyz_id_tensor.view(-1).long()  # [B * max_N]
        # 使用索引从 x 中获取邻居特征
        gathered_feats = x[flat_idx]  # [B * max_N, feat_dim]
        # 恢复形状为 [B, max_N, feat_dim]
        neighbor_feats = gathered_feats.view(xyz_id_tensor.shape[0], xyz_id_tensor.shape[1], -1)
        return neighbor_feats

    def forward(self, x, window_ij_t, current_xyz_nb, current_xyz_id):
        """
        x: [B, feat_dim] 主序列特征张量
        window_ij_t: [B, 1, K] 邻居权重张量   print(current_xyz_nb.shape)
        print(window_ij_t.shape)  这俩打印了还真符合这个
        current_xyz_nb: [B, K, 3] 投影坐标张量（已batched）
        """
        B, _, K = window_ij_t.shape  # 获取批次大小和邻居数量
        # B, _, K = current_xyz_nb.shape  # 获取批次大小和邻居数量
        # print(current_xyz_id)
        # Residual 预处理
        x = x.squeeze()  # [B, feat_dim]
        # print(x.shape)
        residual = self.dropout(self.fc(x))  # [B, Hidden]
        neighbor_features = self.extract_features(x, current_xyz_id)
        # 获取邻居特征并进行处理
        f_j = self.net_in0(neighbor_features)  # [B, K, feat_dim] 处理邻居特征
        # 卷积后的 P_ij：对邻居坐标进行卷积处理
        P_ij = self.conv[0](current_xyz_nb)  # [B, K, 3] 投影坐标经过卷积处理
        # 拼接坐标和特征：将坐标和特征乘积拼接
        x_f = P_ij[:, :, 0:1] * f_j  # [B, K, feat_dim]
        y_f = P_ij[:, :, 1:2] * f_j  # [B, K, feat_dim]
        z_f = P_ij[:, :, 2:3] * f_j  # [B, K, feat_dim]
        concat = torch.cat([x_f, y_f, z_f], dim=-1)  # [B, K, 3*feat_dim]
        # MLP处理：经过MLP变换，确保输入维度是 3 * feat_dim
        transformed = self.mlp(concat.view(B * K, -1))  # [B * K, Hidden]
        # Reshape back to [B, K, Hidden]
        transformed = transformed.view(B, K, -1)
        # 加权求和：通过 window_ij_t 权重对变换后的特征进行加权
        weighted_sum = (window_ij_t.transpose(1, 2) * transformed).sum(dim=1)  # [B, Hidden]
        # BN + Linear：后续的批量归一化和线性变换
        F_ij_tensor1 = self.BN[0](weighted_sum)  # [B, Hidden]
        linear_output = self.linear_layers[0](F_ij_tensor1)  # [B, Hidden]
        # GCN-like 残差模块
        layer_inner = self.act_fn(self.chen[0](linear_output, residual, self.lamda, self.alpha, 1))
        # 输出层
        output = self.net_out(layer_inner)  # [B, out_dim]
        return output.squeeze()  # [B] 或者 [B, out_dim]
