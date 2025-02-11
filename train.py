import os
import time
import sys

import random
import torch
import argparse
import numpy as np
from torch import nn
from model import define_model
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch import optim
from torch.optim import lr_scheduler
import model
import logging
import pandas as pd
from model import ConvLSTM
from torch.nn import functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--lr-start', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-file', type=str, default='./')
    parser.add_argument('--minibatch-size', type=int, default=16)
    parser.add_argument('--margin_p', type=float, default=1)
    parser.add_argument('--margin_t', type=float, default=0.5)
    parser.add_argument('--margin_s', type=float, default=0.5)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--model-name', type=str, default='')
    parser.add_argument('--model-folder', type=str, default='./models')
    parser.add_argument('--discountfactor-p', type=float, default=1)
    parser.add_argument('--discountfactor-t', type=float, default=1)
    parser.add_argument('--discountfactor-s', type=float, default=1)
    parser.add_argument('--window-size-positive', type=float, default=0.35)
    parser.add_argument('--window-size-negative', type=float, default=0.45)
    parser.add_argument('--eff-positive', type=float, default=0.5)
    parser.add_argument('--eff-negative', type=float, default=0.55)

    return parser.parse_args()


def sample_index(data):
    len_seq = data.size(1)
    arange1 = np.arange(0, len_seq)
    tindex = np.random.choice(arange1)
    len_sp = data.size()[2]
    rdm_space = np.arange(0, len_sp)
    sindex = np.random.choice(rdm_space)
    return tindex, sindex

def pair_positive_index(data, tindex):
    positive_tindex = tindex
    return positive_tindex

def pair_negative_index(inflow, demand, tindex, sindex, arguments):
    len_data = inflow.size(1)
    lower_bound = 0
    upper_bound = max(0, tindex - 3)
    negtive_range1 = np.arange(lower_bound, upper_bound)
    lower_bound = min(tindex + 3, len_data)
    upper_bound = len_data
    negtive_range2 = np.arange(lower_bound, upper_bound)
    if len(negtive_range1) == 0:
        negtive_list = negtive_range2
    elif len(negtive_range2) == 0:
        negtive_list = negtive_range1
    else:
        negtive_list = np.concatenate([negtive_range1, negtive_range2])

    arange = np.arange(0, len_data)

    while True:
        inflow_tindex = np.random.choice(arange)
        demand_tindex = np.random.choice(arange)
        if inflow_tindex in negtive_list or demand_tindex in negtive_list:
            break

    len_map = inflow.size(2)

    while True:
        demand_sindex = np.random.randint(0, len_map)
        inflow_sindex = np.random.randint(0, len_map)
        if demand_sindex == sindex and inflow_sindex == sindex:
            continue
        break

    return inflow_tindex, demand_tindex, inflow_sindex, demand_sindex

def pairing_sample(speed, inflow, demand, arguments):
    anchor_tindex, anchor_sindex = sample_index(speed)
    positive_tindex = pair_positive_index(speed, anchor_tindex)
    inflow_negtindex, demand_negtindex, inflow_negsindex, demand_negsindex = pair_negative_index(inflow, demand, anchor_tindex, anchor_sindex, arguments)

    anchor_frame = speed[:, anchor_tindex, anchor_sindex, :, :, :]
    inflow_tpositive = inflow[:, anchor_tindex, anchor_sindex, :, :, :]
    demand_tpositive = demand[:, anchor_tindex, anchor_sindex, :, :, :]

    inflow_neg1 = inflow[:, inflow_negtindex, anchor_sindex, :, :, :]
    demand_neg1 = demand[:, demand_negtindex, anchor_sindex, :, :, :]

    inflow_neg2 = inflow[:, anchor_tindex, inflow_negsindex, :, :, :]
    demand_neg2 = demand[:, anchor_tindex, demand_negsindex, :, :, :]

    return anchor_frame, inflow_tpositive, demand_tpositive, inflow_neg1, demand_neg1, inflow_neg2, demand_neg2

def calculate_dtw(series1, series2):
    if isinstance(series1, torch.Tensor):
        series1_flat = series1.detach().cpu().numpy().flatten()
    else:
        series1_flat = series1.flatten()

    if isinstance(series2, torch.Tensor):
        series2_flat = series2.detach().cpu().numpy().flatten()
    else:
        series2_flat = series2.flatten()

    total_distance, path = fastdtw(series1_flat, series2_flat, dist=2)
    average_distance = total_distance / len(path)

    return average_distance

def t_positive_index_dtw(data, tindex, sindex, arguments):
    len_data = data.size(1)
    anchor_series = data[:, tindex, :, :, :].reshape(-1).detach().cpu().numpy()
    indices = np.random.permutation(len_data)
    min_distance = float('inf')
    positive_tindex = tindex

    for i in indices:
        if i == tindex:
            continue
        compare_series = data[:, i, :, :, :].reshape(-1).detach().cpu().numpy()
        distance = calculate_dtw(anchor_series, compare_series)

        if distance < arguments.window_size_positive:
            return i

        if distance < min_distance:
            min_distance = distance
            positive_tindex = i

    return positive_tindex

def t_neg_index_dtw(data, tindex, sindex, arguments):
    len_data = data.size(1)
    anchor_series = data[:, tindex, :, :, :].reshape(-1).detach().cpu().numpy()
    indices = np.random.permutation(len_data)
    max_distance = float('-inf')
    neg_tindex = tindex

    for i in indices:
        if i == tindex:
            continue
        compare_series = data[:, i, :, :, :].reshape(-1).detach().cpu().numpy()
        distance = calculate_dtw(anchor_series, compare_series)

        if distance > arguments.window_size_negative:
            return i

        if distance > max_distance:
            max_distance = distance
            neg_tindex = i

    return neg_tindex

def temporal_sample(speed_emb, inflow_emb, demand_emb, raw_speed, arguments):
    t_index, s_index = sample_index(speed_emb)

    positive_tindex = t_positive_index_dtw(speed_emb, t_index, s_index, arguments)
    neg_tindex = t_neg_index_dtw(speed_emb, t_index, s_index, arguments)

    speed_anchor = speed_emb[:, t_index, s_index, :, :, :]
    inflow_anchor = inflow_emb[:, t_index, s_index, :, :, :]
    demand_anchor = demand_emb[:, t_index, s_index, :, :, :]

    speed_positive = speed_emb[:, positive_tindex, s_index, :, :, :]
    inflow_positive = inflow_emb[:, positive_tindex, s_index, :, :, :]
    demand_positive = demand_emb[:, positive_tindex, s_index, :, :, :]

    speed_neg = speed_emb[:, neg_tindex, s_index, :, :, :]
    inflow_neg = inflow_emb[:, neg_tindex, s_index, :, :, :]
    demand_neg = demand_emb[:, neg_tindex, s_index, :, :, :]

    return speed_anchor, speed_positive, speed_neg, inflow_anchor, inflow_positive, inflow_neg, demand_anchor, demand_positive, demand_neg




def safe_corrcoef(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    return np.corrcoef(x, y)[0, 1]

def s_positive_index(data, tindex, sindex, arguments):
    len_data = data.size(2)
    anchor_series = data[:, :, sindex, :, :].reshape(-1).detach().cpu().numpy()
    indices = np.random.permutation(len_data)

    for i in indices:
        if i == sindex:
            continue
        compare_series = data[:, :, i, :, :].reshape(-1).detach().cpu().numpy()
        coefficient = safe_corrcoef(anchor_series, compare_series)

        if coefficient > arguments.eff_positive:
            return i

    return sindex

def s_neg_index(data, tindex, sindex, arguments):
    len_data = data.size(2)
    anchor_series = data[:, :, sindex, :, :].reshape(-1).detach().cpu().numpy()
    indices = np.random.permutation(len_data)

    for i in indices:
        if i == sindex:
            continue
        compare_series = data[:, :, i, :, :].reshape(-1).detach().cpu().numpy()
        coefficient = safe_corrcoef(anchor_series, compare_series)

        if coefficient < arguments.eff_negative:
            return i

    return sindex

def spatial_sample(speed_emb, inflow_emb, demand_emb, raw_speed, arguments):
    t_index, s_index = sample_index(speed_emb)

    positive_sindex = s_positive_index(raw_speed, t_index, s_index, arguments)
    neg_sindex = s_neg_index(raw_speed, t_index, s_index, arguments)

    speed_anchor = speed_emb[:, t_index, s_index, :, :, :]
    inflow_anchor = inflow_emb[:, t_index, s_index, :, :, :]
    demand_anchor = demand_emb[:, t_index, s_index, :, :, :]

    speed_positive = speed_emb[:, t_index, positive_sindex, :, :, :]
    inflow_positive = inflow_emb[:, t_index, positive_sindex, :, :, :]
    demand_positive = demand_emb[:, t_index, positive_sindex, :, :, :]

    speed_neg = speed_emb[:, t_index, neg_sindex, :, :, :]
    inflow_neg = inflow_emb[:, t_index, neg_sindex, :, :, :]
    demand_neg = demand_emb[:, t_index, neg_sindex, :, :, :]

    return speed_anchor, speed_positive, speed_neg, inflow_anchor, inflow_positive, inflow_neg, demand_anchor, demand_positive, demand_neg



def trainby_convlstm_1(dataset, city, dataset_name, device):
    dim = dataset.size(1)
    map_num = dataset.size(2)

    model = ConvLSTM(
        input_dim=1,
        hidden_dim=16,
        kernel_size=(3, 3),
        num_layers=3,
        batch_first=True,
        predict_channel=1,
    ).to(device)

    model = model.to(device)

    model_file = f''
    print(f"Loading model from: {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=device))

    model.eval()
    res = []
    for i in range(map_num):
        temp = dataset[:, :, i, :, :]
        temp = temp.unsqueeze(2)
        temp = temp.float().to(device)
        output, state, _ = model(temp)
        res.append(output[-1].detach())
    resx = torch.stack(res, dim=2)
    print(resx.shape)
    return resx

def process_small(speed, demand, inflow):
    """
    Process speed, demand, and inflow data using thresholding and normalization.
    """
    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)

    speed = torch.clamp(speed, max=140)  # Speed thresholding
    demand = torch.clamp(demand, max=demand_threshold)  # Demand thresholding
    inflow = torch.clamp(inflow, max=inflow_threshold)  # Inflow thresholding

    # Normalize each data type
    def normalize(data):
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val == 0:
            return torch.zeros_like(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1

    normalized_speed = normalize(speed).float()
    normalized_demand = normalize(demand).float()
    normalized_inflow = normalize(inflow).float()

    return normalized_speed, normalized_demand, normalized_inflow

class Logger(object):
    def __init__(self, logfilename):
        logging.basicConfig(filename=logfilename, level=logging.DEBUG, filemode='a')

    def info(self, *arguments):
        print(*arguments)
        message = " ".join(map(repr, arguments))
        logging.info(message)


def distance(x1, x2):
    dis = torch.norm(x1 - x2)
    return dis

def tri_distance(x1,x2,x3):
    dis = distance(x1,x2) + distance(x1,x3) + distance(x2,x3)
    return dis

def save_model(model, filename, model_folder):
    model_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), model_path)


def model_filename(model_name, epoch):
    return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)


def unsqueeze(data1, data2):
    data1 = data1.unsqueeze(1)
    data2 = data2.unsqueeze(1)

    return data1, data2


class CustomDataset(Dataset):
    def __init__(self, datasetA, datasetB, datasetC,datasetD):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC
        self.datasetD = datasetD

    def __len__(self):
        return len(self.datasetA)

    def __getitem__(self, idx):
        dataA = self.datasetA[idx]
        dataB = self.datasetB[idx]
        dataC = self.datasetC[idx]
        dataD = self.datasetD[idx]

        return dataA, dataB, dataC, dataD

def find_idle_gpu():
    load = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
    return load.index(min(load))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    arguments = get_args()

    cities = ["XA"]
    data_paths = {
        "XA": {
            "speed": "../data/speed_XA.npy",
            "inflow": "../data/inflow_XA.npy",
            "demand": "../data/demand_XA.npy",
        }
    }

    ST = define_model()
    ST = nn.DataParallel(ST)
    ST.to(device)

    optimizer = optim.Adam(ST.parameters(), lr=arguments.lr_start)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 1000], gamma=0.1)

    for city in cities:
        print(f"Processing city: {city}")

        speed = torch.tensor(np.load(data_paths[city]['speed'])).float()
        inflow = torch.tensor(np.load(data_paths[city]['inflow'])).float()
        demand = torch.tensor(np.load(data_paths[city]['demand'])).float()

        speed = speed[:, :10, :7, :, :]
        inflow = inflow[:, :10, :7, :, :]
        demand = demand[:, :10, :7, :, :]

        raw_speed, raw_demand, raw_inflow = process_small(speed, demand, inflow)

        print(f"Generating embeddings for city: {city}")
        speed_emb = trainby_convlstm_1(raw_speed, city, 'speed', device)
        inflow_emb = trainby_convlstm_1(raw_inflow, city, 'inflow', device)
        demand_emb = trainby_convlstm_1(raw_demand, city, 'demand', device)

        dataset = CustomDataset(speed_emb, inflow_emb, demand_emb, raw_speed)
        data_loader = DataLoader(dataset=dataset, batch_size=arguments.minibatch_size, shuffle=True, pin_memory=False)

        for epoch in range(arguments.start_epoch, arguments.start_epoch + arguments.epochs):
            pairing_loss_sum = 0
            temporal_loss_sum = 0
            spatial_loss_sum = 0
            total_loss_sum = 0
            num_batches = 0

            start_time = time.time()

            for speed, inflow, demand, raw_speed_batch in data_loader:
                total_loss = 0

                anchor_frame, inflow_tpositive, demand_tpositive, inflow_neg1, demand_neg1, inflow_neg2, demand_neg2 = pairing_sample(
                    speed, inflow, demand, arguments)

                anchor_output, inflow_p, demand_p = ST(speed_input=anchor_frame, inflow_input=inflow_tpositive,
                                                       demand_input=demand_tpositive)

                _, inflow_n1, _ = ST(inflow_input=inflow_neg1)
                _, _, demand_n1 = ST(demand_input=demand_neg1)
                _, inflow_n2, _ = ST(inflow_input=inflow_neg2)
                _, _, demand_n2 = ST(demand_input=demand_neg2)

                d_t_positive = tri_distance(anchor_output, inflow_p, demand_p)
                d_t_negative1 = tri_distance(anchor_output, inflow_n1, demand_n1)
                d_t_negative2 = tri_distance(anchor_output, inflow_n2, demand_n2)

                pair_positve = d_t_positive / (d_t_positive + d_t_negative1 + d_t_negative2)
                pair_neg1 = d_t_negative1 / (d_t_negative1 + d_t_negative2 + d_t_positive)
                pair_neg2 = d_t_negative2 / (d_t_negative1 + d_t_negative2 + d_t_positive)

                pairing = arguments.discountfactor_p * torch.clamp(
                    arguments.margin_p + pair_positve - pair_neg1 - pair_neg2, min=0).mean()

                total_loss += pairing

                speed_anchor, speed_positive, speed_neg, inflow_anchor, inflow_positive, inflow_neg, demand_anchor, demand_positive, demand_neg = temporal_sample(
                    speed, inflow, demand, raw_speed_batch, arguments)

                anchor_tspeed, anchor_tinflow, anchor_tdemand = ST(speed_input=speed_anchor,
                                                                   inflow_input=inflow_anchor, demand_input=demand_anchor)

                positive_tspeed, positive_tinflow, positive_tdemand = ST(speed_input=speed_positive,
                                                                         inflow_input=inflow_positive,
                                                                         demand_input=demand_positive)

                neg_tspeed, neg_tinflow, neg_tdemand = ST(speed_input=speed_neg,
                                                          inflow_input=inflow_neg, demand_input=demand_neg)

                tpair_positive = distance(anchor_tspeed, positive_tspeed) + distance(anchor_tinflow, positive_tinflow) + distance(anchor_tdemand, positive_tdemand)
                tpair_neg = distance(anchor_tspeed, neg_tspeed) + distance(anchor_tinflow, neg_tinflow) + distance(anchor_tdemand, neg_tdemand)

                temporal = arguments.discountfactor_t * torch.clamp(arguments.margin_t + tpair_positive - tpair_neg, min=0).mean()
                total_loss += temporal

                speed_anchor, speed_positive, speed_neg, inflow_anchor, inflow_positive, inflow_neg, demand_anchor, demand_positive, demand_neg = spatial_sample(
                    speed, inflow, demand, raw_speed_batch, arguments)

                anchor_tspeed, anchor_tinflow, anchor_tdemand = ST(speed_input=speed_anchor,
                                                                   inflow_input=inflow_anchor, demand_input=demand_anchor)

                positive_tspeed, positive_tinflow, positive_tdemand = ST(speed_input=speed_positive,
                                                                         inflow_input=inflow_positive,
                                                                         demand_input=demand_positive)

                neg_tspeed, neg_tinflow, neg_tdemand = ST(speed_input=speed_neg,
                                                          inflow_input=inflow_neg, demand_input=demand_neg)

                spair_positive = distance(anchor_tspeed, positive_tspeed) + distance(anchor_tinflow, positive_tinflow) + distance(anchor_tdemand, positive_tdemand)
                spair_neg = distance(anchor_tspeed, neg_tspeed) + distance(anchor_tinflow, neg_tinflow) + distance(anchor_tdemand, neg_tdemand)

                spatial = arguments.discountfactor_s * torch.clamp(arguments.margin_s + spair_positive - spair_neg, min=0).mean()
                total_loss += spatial

                pairing_loss_sum += pairing.item()
                temporal_loss_sum += temporal.item()
                spatial_loss_sum += spatial.item()
                total_loss_sum += total_loss.item()

                num_batches += 1

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            scheduler.step()

            if epoch % arguments.save_every == 0 and epoch != 0:
                model_name = f""
                save_model(ST, model_name, arguments.model_folder)

if __name__ == '__main__':
    main()

