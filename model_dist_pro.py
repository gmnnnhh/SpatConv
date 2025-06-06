import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import numpy as np

dropout_rate = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0, lamda, alpha, l):
        theta = min(1, math.log(lamda / l + 1))
        # hi = torch.spmm(adj, input)
        hi = input
        # print(f"hi shape: {hi.shape}, h0 shape: {h0.shape}")

        support = (1 - alpha) * hi + alpha * h0
        r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r  # GCNII
        output = output + input
        return output


class Spatom(nn.Module):
    '''
    changed from https://github.com/chennnM/GCNII
    '''

    def __init__(self, in_channels=1024, out_channels=1, n_layers=3, lamda=LAMBDA,
                 alpha=ALPHA, sigma=14):
        super(Spatom, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Input = in_channels
        self.Output = out_channels

        self.Hidden = 64
        self.Cuts = 16

        self.fc = nn.Linear(in_channels, self.Hidden)

        self.BN = nn.ModuleList(
            [nn.BatchNorm1d(self.Hidden) for iteration in range(n_layers)])
        self.dropout = nn.Dropout(p=0.2)

        self.net_in = nn.ModuleList([nn.Sequential(
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2), ) for iteration in range(n_layers)])

        self.net_in0 = nn.Sequential(
            nn.Linear(self.Input, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),  # 也加入 dropout了 0.5
            nn.Linear(self.Hidden, self.Hidden),
            nn.LeakyReLU(negative_slope=0.2), )  #  (H,)

        self.conv = nn.ModuleList([nn.Sequential(
            nn.Linear(3, self.Cuts),
            nn.ReLU(),  # KeOps does not support well LeakyReLu

            nn.Linear(self.Cuts, self.Hidden),
        ) for iteration in range(n_layers)])

        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Output, self.Output),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),
            nn.Sigmoid()
        )
        self.chen = nn.ModuleList(
            [GraphChenn(self.Hidden, self.Hidden) for iteration in range(n_layers)])

        self.alpha = alpha
        self.lamda = lamda
        self.sigma = sigma

        with torch.no_grad():
            for seq_module in self.conv:
                nn.init.normal_(seq_module[0].weight)
                nn.init.uniform_(seq_module[0].bias)
                seq_module[0].bias *= 0.8 * (seq_module[0].weight ** 2).sum(-1).sqrt()

                nn.init.uniform_(seq_module[2].weight, a=-1 / np.sqrt(self.Cuts), b=1 / np.sqrt(self.Cuts))
                nn.init.normal_(seq_module[2].bias)
                seq_module[2].bias *= 0.5 * (seq_module[2].weight ** 2).sum(-1).sqrt()

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.Hidden, self.Hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
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
        self.distance_thresholds = [13, 13, 13]

    def print_hyperparameters_and_architecture(self):

        print("Hyperparameters:")
        hyperparams = ['Input', 'Output', 'Hidden', 'Cuts', 'alpha', 'lamda', 'sigma', 'distance_thresholds']
        for param in hyperparams:
            print(f'{param} = {getattr(self, param)}')

        print("\nModel Architecture:")
        print(self)

    def extract_features(self, x, xyz_id):

        neighbor_features = {}
        for key, selected_neighbors in xyz_id.items():
            neighbor_feats = x[torch.stack(selected_neighbors)]

            neighbor_features[key] = neighbor_feats
        return neighbor_features

    def forward(self, x, xyz_nb, xyz_id, dij, ):

        x = x.squeeze()

        j = 0
        previous_outputs = []

        for threshold in self.distance_thresholds:

            current_xyz_id = {}
            current_xyz_nb = {}
            current_dij = {}

            for key, distances in dij.items():
                euclidean_distances = torch.sqrt(torch.sum(distances ** 2, axis=1))
                valid_indices = torch.arange(len(euclidean_distances))[euclidean_distances < threshold]

                current_xyz_id[key] = [xyz_id[key][i] for i in valid_indices]
                current_xyz_nb[key] = [xyz_nb[key][i] for i in valid_indices]
                current_dij[key] = [dij[key][i] for i in valid_indices]
            neighbor_features = self.extract_features(x, current_xyz_id)

            if j == 0:
                residual = self.dropout(self.fc(x))
                f_j = {key: self.net_in0(value) for key, value in neighbor_features.items()}
            else:
                f_j = {key: self.net_in[j](value) for key, value in neighbor_features.items()}

            # P_ij = {key: self.conv[j](value) for key, value in current_xyz_nb.items()}
            P_ij = {}
            for key, value in current_xyz_nb.items():

                if isinstance(value, list):
                    value_tensor = torch.stack(value).to(self.device)
                else:
                    value_tensor = value
                P_ij[key] = self.conv[j](value_tensor)

                # P_ij[key] = value_tensor
            # d2_ij_dict = {}
            window_ij_dict = {}
            window_ij_t_dict = {}
            for key, value in current_dij.items():
                value_tensor = torch.stack(value)
                d2_ij = (value_tensor ** 2).sum(-1)
                window_ij = torch.exp(-d2_ij / (2 * threshold ** 2))
                # d2_ij_dict[key] = d2_ij
                window_ij_dict[key] = window_ij
                window_ij_t_dict[key] = window_ij.unsqueeze(0)

            F_ij_tensor = torch.zeros(len(P_ij), self.Hidden)
            F_ij_tensor = F_ij_tensor.to(DEVICE)

            for i, (key, P_value) in enumerate(P_ij.items()):
                f_j_value = f_j[key]

                x_coord = P_value[:, 0].unsqueeze(1)  # [N, 1]
                y_coord = P_value[:, 1].unsqueeze(1)  # [N, 1]
                z_coord = P_value[:, 2].unsqueeze(1)  # [N, 1]

                x_feature = x_coord * f_j_value
                y_feature = y_coord * f_j_value
                z_feature = z_coord * f_j_value

                concatenated_features = torch.cat((x_feature, y_feature, z_feature), dim=1)  # [N, 3*feature_dim]

                transformed_features = self.mlp(concatenated_features)
                window_ij_t_value = window_ij_t_dict[key]
                F_value = window_ij_t_value @ transformed_features
                F_ij_sum = F_value.sum(dim=0, keepdim=True)
                F_ij_tensor[i] = F_ij_sum

            F_ij_tensor1 = self.BN[j](F_ij_tensor)
            linear_output = self.linear_layers[j](F_ij_tensor1)

            layer_inner = self.act_fn(self.chen[j](linear_output, residual, self.lamda, self.alpha, j + 1))

            x = layer_inner
            j += 1

        output = self.net_out(x)
        return output.squeeze()


