import datetime
import torch.optim.lr_scheduler as lr_scheduler
import torch
import os
import pickle
import numpy as np
from model_dist import SpatConv
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import torch.optim as optim
import time
import sklearn.metrics as skm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)


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
    seed = 784363031
    set_random_seed(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

LEARN_RATE = 0.001
EPOCH = 20
train_set = pickle.load(
    open('/mnt/storage1/guanmm/New/customed_data/PP/feature/' + 'train' + '_feature.pkl', 'rb'))
test_set = pickle.load(
    open('/mnt/storage1/guanmm/New/customed_data/PP/feature/' + 'test' + '_feature.pkl', 'rb'))


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

seed = 784363031


def train(train_set=train_set):
    samples_num = len(train_set)
    # print(train_set)
    split_num = int(70 / 85 * samples_num)
    data_index = np.arange(samples_num)
    # np.random.seed(1205)
    # np.random.shuffle(data_index)
    # seed = get_new_seed()
    # save_seed(seed, 'seeds.txt')
    set_random_seed(seed)
    np.random.shuffle(data_index)
    # set_random_seed(seed)

    train_index = data_index[:split_num]
    valid_index = data_index[split_num:]
    train_loader = DataLoader(train_set, batch_size=1, sampler=train_index)
    valid_loader = DataLoader(train_set, batch_size=1, sampler=valid_index)
    model = SpatConv()
    model.print_hyperparameters_and_architecture()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=weight_decay)
    loss_fun = nn.BCELoss()

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

    patience = 5
    counter = 0
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
            counter = 0
        else:
            counter += 1
        scheduler.step(f_1)

        if torch.cuda.is_available():
            model.cuda()

        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    plt.plot(train_log, 'r-', label='Training Loss')
    plt.plot(valid_log, 'b-', label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{result_path}/train_valid_loss.png")
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
    # data_dict = {}
    skipped_count = 0
    for step, data in enumerate(train_loader):
        # feature = data.x.to(DEVICE, dtype=torch.float)
        prefea_array = np.array(data.prefea)
        prefea = torch.tensor(prefea_array).to(DEVICE, dtype=torch.float)
        # name = data.name[0]
        label = data.y.to(DEVICE, dtype=torch.float)
        xyz_nb = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.xyz_nb.items()}
        xyz_id = {key: tensor.to(DEVICE, dtype=torch.long) for key, tensor in data.xyz_id.items()}
        dij = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.dij.items()}

        optimizer.zero_grad()
        pred, x = model(prefea, xyz_nb, xyz_id, dij)
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
    skipped_count = 0
    with torch.no_grad():
        for step, data in enumerate(valid_loader):
            prefea_array = np.array(data.prefea)
            prefea = torch.tensor(prefea_array).to(DEVICE, dtype=torch.float)
            label = data.y.to(DEVICE, dtype=torch.float)
            xyz_nb = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.xyz_nb.items()}
            xyz_id = {key: tensor.to(DEVICE, dtype=torch.long) for key, tensor in data.xyz_id.items()}
            dij = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.dij.items()}
            pred, _ = model(prefea, xyz_nb, xyz_id, dij)
            valid_loss = loss_fun(pred, label)
            pred = pred.cpu().numpy()
            all_label.extend(label.cpu().numpy())
            all_pred.extend(pred)
            loss = loss + valid_loss.item()
            num += 1

    epoch_loss = loss / num
    accuracy, recall, precision, MCC, f_1_max, t_max = best_f_1(np.array(all_label), np.array(all_pred))

    return epoch_loss, accuracy, recall, precision, MCC, f_1_max, t_max


def test(test_set):
    test_loader = DataLoader(test_set, batch_size=1)
    model = SpatConv().to(DEVICE)
    model.load_state_dict(torch.load(f'{result_path}/best_model.dat'))
    model.eval()
    all_label = []
    all_pred = []
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            prefea_array = np.array(data.prefea)
            prefea = torch.tensor(prefea_array).to(DEVICE, dtype=torch.float)
            label = data.y.to(DEVICE, dtype=torch.float)
            xyz_nb = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.xyz_nb.items()}
            xyz_id = {key: tensor.to(DEVICE, dtype=torch.long) for key, tensor in data.xyz_id.items()}
            dij = {key: tensor.to(DEVICE, dtype=torch.float) for key, tensor in data.dij.items()}
            pos = data.POS[0]
            pos1 = data.POS1[0]
            pos1_set = set(pos1)
            length = data.length.item()

            pred, x = model(prefea, xyz_nb, xyz_id, dij)
            pred = pred.cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            predict_protein = [0] * length
            for k, i in enumerate(pos):
                if i not in pos1_set:
                    predict_protein[i] = pred[k]
            all_label.extend(label)
            all_pred.extend(predict_protein)

    accuracy, recall, precision, MCC, f_1, t_max = best_f_1(np.array(all_label), np.array(all_pred))
    AUC = skm.roc_auc_score(all_label, all_pred)
    precisions, recalls, thresholds = skm.precision_recall_curve(all_label, all_pred)
    AUPRC = skm.auc(recalls, precisions)
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
