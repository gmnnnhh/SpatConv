import datetime
import torch.optim.lr_scheduler as lr_scheduler
import torch
import os
import pickle
import numpy as np
from model import SpatConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler
import torch.optim as optim
import time
import sklearn.metrics as skm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 随着性能改变的学习率
import random
import pandas as pd
# 创建 TensorBoard 摘要编写器

# torch.manual_seed(1209)
# np.random.seed(1205)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1209)
#     torch.cuda.set_device(0)
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)  # 设置CUDA设备，如果有多个设备可以更改此部分


def get_new_seed():
    return random.randint(0, 2 ** 32 - 1)


def save_seed(seed, file_path):
    with open(file_path, 'a') as file:
        file.write(f'{seed}\n')


def load_seeds(file_path):
    with open(file_path, 'r') as file:
        seeds = [int(line.strip()) for line in file]
    return seeds


def main():
    seeds_file = 'seeds.txt'

    # Check if seeds file exists, otherwise create it
    if not os.path.exists(seeds_file):
        with open(seeds_file, 'w') as file:
            file.write('')

    # Generate a new seed
    seed = get_new_seed()

    # Save the new seed to the file
    save_seed(seed, seeds_file)

    # Set the random seed for the current run
    set_random_seed(seed)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

LEARN_RATE = 0.01  # 改了 0.001
EPOCH = 50  # 改了 150
train_set = pickle.load(open('/mnt/data0/uploaduser/' + 'train' + '_feature_train_dist0.pkl', 'rb'))
test_set = pickle.load(open('/mnt/data0/uploaduser/' + 'test' + '_feature_train_dist0.pkl', 'rb'))  # _13rsa  _yuce


def best_f_1(label, output):
    f_1_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100
        predict = np.where(output > threshold, 1, 0)
        f_1 = skm.f1_score(label, predict, pos_label=1)
        if f_1 > f_1_max:
            f_1_max = f_1
            t_max = threshold

    pred = np.where(output > t_max, 1, 0)
    accuracy = skm.accuracy_score(label, pred)
    recall = skm.recall_score(label, pred)
    precision = skm.precision_score(label, pred)
    MCC = skm.matthews_corrcoef(label, pred)
    return accuracy, recall, precision, MCC, f_1_max, t_max


weight_decay = 1e-5


def train(train_set=train_set):
    # samples_num = len(train_set)
    # # print(train_set)
    # split_num = int(70 / 85 * samples_num)
    # data_index = np.arange(samples_num)
    # np.random.seed(1205)
    # np.random.shuffle(data_index)
    # train_index = data_index[:split_num]      #！！！！！！！！！！

    samples_num = len(train_set)
    # print(train_set)
    split_num = int(70 / 85 * samples_num)
    data_index = np.arange(samples_num)
    # np.random.seed(1205)
    # np.random.shuffle(data_index)
    seed = get_new_seed()
    save_seed(seed, 'seeds.txt')
    set_random_seed(seed)
    np.random.shuffle(data_index)
    # # set_random_seed(seed)
    train_index = data_index[:samples_num]  # 原来所有样本
    valid_index = data_index[split_num:]

    # 限制数据集的样本数（例如，只使用前100个样本进行测试）
    # small_sample_size = 10  # 你可以根据需要修改这个数量
    # train_index = data_index[:split_num][:small_sample_size]  # 只选择小样本量
    # valid_index = data_index[split_num:][:small_sample_size]  # 只选择小样本量

    # 使用SubsetRandomSampler从train_set中根据索引抽取样本
    # train_sampler = SubsetRandomSampler(train_index)
    # valid_sampler = SubsetRandomSampler(valid_index)

    train_loader = DataLoader(train_set, batch_size=1, sampler=train_index)
    valid_loader = DataLoader(train_set, batch_size=1, sampler=valid_index)
    model = SpatConv()
    # model.print_hyperparameters_and_architecture()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=weight_decay)
    loss_fun = nn.BCELoss()  # 二分类交叉熵损失函数

    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)

    train_log = []
    valid_log = []
    valid_f_1 = []
    max_f_1 = 0
    max_acc = 0
    max_precision = 0
    max_recall = 0
    max_epoch = 0
    max_MCC = 0
    max_t = 0

    patience = 4  # 设置早停的耐心值，即允许性能没有改善的最大轮数
    counter = 0  # 初始化计数器
    for epoch in range(EPOCH):
        start_time = time.time()
        loss = train_epoch(model, train_loader, optimizer, loss_fun)
        train_log.append(loss)
        loss_v, accuracy, recall, precision, MCC, f_1, t_max = valid_epoch(model, valid_loader, loss_fun)
        end_time = time.time()
        valid_log.append(loss_v)
        valid_f_1.append(f_1)

        print("Epoch: ", epoch + 1, "|", "Epoch Time: ", end_time - start_time, "s")
        print("Train loss: ", loss)
        print("valid loss: ", loss_v)
        print('F_1:', f_1, t_max)
        print('ACC:', accuracy)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('MCC: ', MCC)

        if f_1 > max_f_1:
            max_f_1 = f_1
            max_acc = accuracy
            max_precision = precision
            max_recall = recall
            max_MCC = MCC
            max_epoch = epoch + 1
            max_t = t_max
            torch.save(model.cpu().state_dict(), f'{result_path}/best_model.dat')
            counter = 0  # 重置计数器
        else:
            counter += 1  # 没有改善，则计数器加1

        scheduler.step(f_1)

        # 将模型移回到原始设备上，如果使用了cuda
        if torch.cuda.is_available():
            model.cuda()

        # 检查是否达到早停条件
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break  # 提前终止训练
            ########
    plt.plot(train_log, 'r-', label='Training Loss')
    plt.plot(valid_log, 'b-', label='Validation Loss')

    # 添加图例
    plt.legend()

    # 添加图表标题和坐标轴标签
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 保存图表
    plt.savefig(f"{result_path}/train_valid_loss.png")

    # 关闭图表，以便释放内存
    plt.close()

    plt.plot(valid_f_1, 'r-')
    plt.title('valid_f_1')
    plt.savefig(f"{result_path}/valid_f_1.png")
    plt.close()

    print("Max Epoch: ", max_epoch)
    print('F_1:', max_f_1, max_t)
    print('ACC:', max_acc)
    print('Precision: ', max_precision)
    print('Recall: ', max_recall)
    print('MCC: ', max_MCC)


def train_epoch(model, train_loader, optimizer, loss_fun):
    model.to(DEVICE)
    model.train()
    loss = 0
    num = 0
    for step, data in enumerate(train_loader):

        # prefea_array = np.array(data.prefea)
        prefea = data.prefea.clone().detach().requires_grad_(True).squeeze().to(DEVICE, dtype=torch.float)
        label = data.y.to(DEVICE, dtype=torch.float)
        window_ij_t_dict = data.window_ij_t.to(DEVICE, dtype=torch.float)

        current_xyz_id = data.xyz_id_tensor.to(DEVICE, dtype=torch.float)
        current_xyz_nb = data.p_ij_tensor.to(DEVICE, dtype=torch.float)


        optimizer.zero_grad()

        pred = model(prefea, window_ij_t_dict, current_xyz_nb, current_xyz_id)
        # pred = model(prefea, window_ij_t_dict,current_xyz_id,p_ij) #prefea, xyz_nb, xyz_id, dij

        train_loss = loss_fun(pred, label)
        train_loss.backward()
        optimizer.step()
        loss = loss + train_loss.item()
        num += 1

    epoch_loss = loss / num
    return epoch_loss


def valid_epoch(model, valid_loader, loss_fun):
    model.to(DEVICE)
    model.eval()
    loss = 0
    num = 0
    all_label = []
    all_pred = []
    with torch.no_grad():
        for step, data in enumerate(valid_loader):

            prefea = data.prefea.clone().detach().squeeze().to(DEVICE, dtype=torch.float)
            label = data.y.to(DEVICE, dtype=torch.float)
            current_xyz_id = data.xyz_id_tensor.to(DEVICE, dtype=torch.float)

            # 只保留这两个：
            window_ij_t_dict = data.window_ij_t.to(DEVICE, dtype=torch.float)

            current_xyz_nb = data.p_ij_tensor.to(DEVICE, dtype=torch.float)

            pred = model(prefea, window_ij_t_dict, current_xyz_nb, current_xyz_id)
            # pred = model(prefea, window_ij_t_dict, current_xyz_id, p_ij)
            valid_loss = loss_fun(pred, label)
            pred = pred.cpu().numpy()
            all_label.extend(label.cpu().numpy())
            all_pred.extend(pred)
            loss = loss + valid_loss.item()
            num += 1


    epoch_loss = loss / num
    accuracy, recall, precision, MCC, f_1_max, t_max = best_f_1(np.array(all_label), np.array(all_pred))
    # scheduler.step(valid_loss / len(valid_loader))   # 加入学习率调度器啦啦 没成功加入哈哈
    return epoch_loss, accuracy, recall, precision, MCC, f_1_max, t_max


def test(test_set):
    start_time = time.time()
    test_loader = DataLoader(test_set, batch_size=1)
    model = SpatConv().to(DEVICE)
    model.load_state_dict(torch.load(f'{result_path}/best_model.dat'))
    model.eval()
    all_label = []
    all_pred = []
    skipped_count = 0

    with torch.no_grad():
        for step, data in enumerate(test_loader):
            # prefea_array = np.array(data.prefea)
            prefea = data.prefea.clone().detach().squeeze().to(DEVICE, dtype=torch.float)
            label = data.y.to(DEVICE, dtype=torch.float)
            window_ij_t_dict = data.window_ij_t.to(DEVICE, dtype=torch.float)
            current_xyz_id = data.xyz_id_tensor.to(DEVICE, dtype=torch.float)
            current_xyz_nb = data.p_ij_tensor.to(DEVICE, dtype=torch.float)

            pos = data.POS[0]
            pos1 = data.POS1[0]
            pos1_set = set(pos1)  # 将 pos1 转换为集合以优化查找操作
            length = data.length.item()

            pred = model(prefea, window_ij_t_dict, current_xyz_nb, current_xyz_id)
            pred = pred.cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            predict_protein = [0] * length
            for k, i in enumerate(pos):
                if i not in pos1_set:
                    predict_protein[i] = pred[k]  # 直接使用pred中的预测值更新predict_protein
            all_label.extend(label)  # 将真实标签添加到all_label列表中
            all_pred.extend(predict_protein)  # 将预测结果添加到all_pred列表中



    accuracy, recall, precision, MCC, f_1, t_max = best_f_1(np.array(all_label), np.array(all_pred))
    AUC = skm.roc_auc_score(all_label, all_pred)
    precisions, recalls, thresholds = skm.precision_recall_curve(all_label, all_pred)
    AUPRC = skm.auc(recalls, precisions)
    end_time = time.time()  # Record end time
    runtime = end_time - start_time  # Calculate runtime in seconds
    print(f"Test Results (Runtime: {runtime:.2f} seconds)\n")
    print("test: ")
    print('F_1:', f_1, t_max)
    print('ACC:', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('MCC: ', MCC)
    print('AUROC: ', AUC)
    print('AUPRC: ', AUPRC)

    return f_1


if __name__ == '__main__':
    path_dir = "./Newresult"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    result_path = f'{path_dir}/{localtime}'
    # result_path = './result/2022-12-05-18:47:00'
    os.makedirs(result_path)
    train()
    test(test_set)