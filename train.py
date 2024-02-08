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
import convlstm.convlstm
from convlstm.convlstm import ConvLSTM
from torch.nn import functional as F


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--lr-start', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--log-file', type=str, default='./')
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--margin_p', type=float, default=0.4)
    parser.add_argument('--margin_t', type=float, default=0.2)
    parser.add_argument('--margin_s', type=float, default=0.2)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--model-name', type=str, default='')
    parser.add_argument('--model-folder', type=str, default='./')
    parser.add_argument('--discountfactor-p', type=float, default=0.99)
    parser.add_argument('--discountfactor-t', type=float, default=0.99)
    parser.add_argument('--discountfactor-s', type=float, default=0.99)
    parser.add_argument('--window-size-positive', type=float, default=1)
    parser.add_argument('--window-size-negative', type=float, default=5)

    return parser.parse_args()


def loaddata(data_name, arguments):
    path = arguments.path + data_name
    data = pd.read_csv(path, header=None, encoding='utf-8', engine='python')
    data = np.array(data)
    data1 = data.reshape(-1, 63, 10, 10)
    return data1

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
    negtive_list = []
    lower_bound = 0
    upper_bound = max(0, tindex - arguments.window_size_negative)
    negtive_range1 = np.arange(lower_bound, upper_bound)

    lower_bound = min(tindex + arguments.window_size_negative, len_data)
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

    return inflow_tindex,demand_tindex,inflow_sindex,demand_sindex


def pairing_sample(speed, inflow, demand ,arguments):
    anchor_tindex, anchor_sindex = sample_index(speed)
    positive_tindex = pair_positive_index(speed , anchor_tindex)
    inflow_negtindex, demand_negtindex, inflow_negsindex, demand_negsindex = pair_negative_index(inflow, demand, anchor_tindex, anchor_sindex,arguments)


    anchor_frame = speed[:, anchor_tindex, anchor_sindex, :, :, :]
    inflow_tpositive = inflow[:,anchor_tindex,anchor_sindex,:,:,:]
    demand_tpositive = demand[:,anchor_tindex,anchor_sindex,:,:,:]


    inflow_neg1 = inflow[:, inflow_negtindex, anchor_sindex, :,:,:]
    demand_neg1 = demand[:, demand_negtindex, anchor_sindex, :,:,:]


    inflow_neg2 = inflow[:, anchor_tindex, inflow_negsindex,:,:,:]
    demand_neg2 = demand[:, anchor_tindex, demand_negsindex,:,:,:]

    return anchor_frame, inflow_tpositive, demand_tpositive, inflow_neg1, demand_neg1, inflow_neg2, demand_neg2

def t_positive_index(data, tindex, arguments):
    len_data = data.size(1)
    positive_tlist = []
    for i in range(1, arguments.window_size_positive + 1):
        positve_tindex1 = tindex + i
        if (positve_tindex1 < len_data):
            positive_tlist.append(positve_tindex1)
        positve_tindex2 = tindex - i
        if (positve_tindex2 >= 0):
            positive_tlist.append(positve_tindex2)
        if (positve_tindex1 >= len_data and positve_tindex2 < 0):
            break
    positive_tindex = np.random.choice(positive_tlist)
    return positive_tindex


def t_neg_index(data, tindex, arguments):
    len_data = data.size(1)
    negtive_list = []
    lower_bound = 0
    upper_bound = max(0, tindex - arguments.window_size_negative)
    negtive_range1 = np.arange(lower_bound, upper_bound)

    lower_bound = min(tindex + arguments.window_size_negative, len_data)
    upper_bound = len_data
    negtive_range2 = np.arange(lower_bound, upper_bound)
    if len(negtive_range1) == 0:
        negtive_list = negtive_range2
    elif len(negtive_range2) == 0:
        negtive_list = negtive_range1
    else:
        negtive_list = np.concatenate([negtive_range1, negtive_range2])

    neg_tindex = np.random.choice(negtive_list)

    return neg_tindex


def temporal_sample(speed, inflow, demand, arguments):
    t_index, s_index = sample_index(speed)
    positive_tindex = t_positive_index(speed, t_index, arguments)
    neg_tindex = t_neg_index(speed, t_index, arguments)


    speed_anchor = speed[:, t_index, s_index, :,:,:]
    inflow_anchor = inflow[:, t_index, s_index, :,:,:]
    demand_anchor = demand[:,t_index,s_index,:,:,:]


    speed_positive = speed[:, positive_tindex, s_index, :,:,:]
    inflow_positive = inflow[:,positive_tindex,s_index,:,:,:]
    demand_positive = demand[:,positive_tindex,s_index,:,:,:]


    speed_neg = speed[:, neg_tindex, s_index, :,:,:]
    inflow_neg = inflow[:,neg_tindex,s_index,:,:,:]
    demand_neg = demand[:,neg_tindex,s_index,:,:,:]

    return speed_anchor,speed_positive,speed_neg,inflow_anchor,inflow_positive,inflow_neg,demand_anchor,demand_positive,demand_neg



def s_positive_index(data, tindex, arguments):
    len_data = data.size(1)
    positive_tlist = []

    for i in range(1, arguments.window_size_positive + 1):
        positve_tindex1 = tindex + i
        if (positve_tindex1 < len_data):
            positive_tlist.append(positve_tindex1)
        positve_tindex2 = tindex - i
        if (positve_tindex2 >= 0):
            positive_tlist.append(positve_tindex2)
        if (positve_tindex1 >= len_data and positve_tindex2 < 0):
            break
    positive_tindex = np.random.choice(positive_tlist)
    return positive_tindex


def s_neg_index(data, sindex):
    len = data.size(2)
    arr = np.arange(0, len)
    while True:
        neg_index = np.random.choice(arr)
        if neg_index == sindex:
            continue
        break
    return neg_index


def spatial_sample(speed, inflow, demand, arguments):
    tindex, sindex = sample_index(speed)
    positive_tindex = s_positive_index(speed, tindex , arguments)
    neg_sindex = s_neg_index(speed, sindex)


    speed_anchor = speed[:,tindex,sindex,:,:,:]
    inflow_anchor = inflow[:,tindex,sindex,:,:,:]
    demand_anchor = demand[:,tindex,sindex,:,:,:]


    speed_positive = speed[:,positive_tindex,sindex,:,:,:]
    inflow_positive = inflow[:,positive_tindex,sindex,:,:,:]
    demand_positive = demand[:,positive_tindex,sindex,:,:,:]


    speed_neg = speed[:,tindex,neg_sindex,:,:,:]
    inflow_neg = inflow[:,tindex,neg_sindex,:,:,:]
    demand_neg = demand[:,tindex,neg_sindex,:,:,:]

    return speed_anchor, speed_positive, speed_neg, inflow_anchor, inflow_positive, inflow_neg, demand_anchor,demand_positive,demand_neg


def trainby_convlstm_1(dataset, dataset_name,device):
    dim = dataset.size(2)
    map_num = dataset.size(3)

    model = ConvLSTM(input_dim=dim,
                     hidden_dim=16,
                     kernel_size=(3, 3),
                     num_layers=3,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False,
                     predict_channel=dim)

    model = nn.DataParallel(model, device_ids=[1,0,2,3])
    model = model.to(device)

    model_file = f''
    model.module.load_state_dict(torch.load(model_file, map_location=device))


    model.eval()
    res = []
    for i in range(map_num):
        temp = dataset[:, :, :, i, :, :]
        temp = temp.float()
        output, state, _ = model(temp)
        res.append(output[0])
    resx = torch.stack(res, dim=2)

    return resx

def process_d(speed, demand, inflow):

    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)

    speed_np = torch.clamp(speed, max=140)
    demand_np = torch.clamp(demand, max=demand_threshold)
    inflow_np = torch.clamp(inflow, max=inflow_threshold)  

    x1 = normalize(demand_np)
    y1 = normalize(inflow_np)
    z1 = normalize(speed_np)

    temp2 = y1.unsqueeze(1)
    temp1 = x1.unsqueeze(1)

    temp3 = z1.unsqueeze(1)
    res_speed = temp3.reshape(-1, 12, 1, 63, 10, 10)
    res_demand = temp1.reshape(-1, 12, 1, 63, 10, 10)
    res_inflow = temp2.reshape(-1, 12, 1, 63, 10, 10)

    return res_speed, res_demand, res_inflow

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


def normalize(data):
    min_val = torch.min(data)
    max_val = torch.max(data)

    if max_val - min_val == 0:
        return torch.zeros_like(data)

    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1

    return normalized_data


class CustomDataset(Dataset):
    def __init__(self, datasetA, datasetB, datasetC):

        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC

    def __len__(self):
        return len(self.datasetA)

    def __getitem__(self, idx):

        dataA = self.datasetA[idx]
        dataB = self.datasetB[idx]
        dataC = self.datasetC[idx]


        return dataA, dataB, dataC

def find_idle_gpu():
    load = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
    return load.index(min(load))


def main():
    
    idle_gpu = find_idle_gpu()
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    arguments = get_args()
    logger = Logger(arguments.log_file)


    use_cuda = torch.cuda.is_available()
    ST = define_model()
    ST = nn.DataParallel(ST,device_ids=[1,0,2,3])
    ST.to(device)

    speed = loaddata('../speed_SZ.csv', arguments)
    inflow = loaddata('../inflow_SZ.csv', arguments)
    demand = loaddata('../demand_SZ.csv', arguments)

    speed = torch.tensor(speed).to(device)
    inflow = torch.tensor(inflow).to(device)
    demand = torch.tensor(demand).to(device)

    speed, demand, inflow = process_d(speed, demand, inflow)


    optimizer = optim.Adam(ST.parameters(), lr=arguments.lr_start)
    learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 1000], gamma=0.1)

    res_speed1 = trainby_convlstm_1(speed, 'speed',device)
    res_inflow1 = trainby_convlstm_1(inflow,'inflow',device)
    res_demand1 = trainby_convlstm_1(demand,'demand',device)

    dataset = CustomDataset(res_speed1, res_inflow1, res_demand1)
    data_loader = DataLoader(dataset=dataset, batch_size=arguments.minibatch_size, shuffle=True, pin_memory=False)

    for epoch in range(arguments.start_epoch, arguments.start_epoch + arguments.epochs):
        pairing_loss_sum = 0
        temporal_loss_sum = 0
        spatial_loss_sum = 0
        total_loss_sum = 0
        num_batches = 0


        start_time = time.time()

        for speed , inflow, demand in data_loader:

            speed = speed
            inflow = inflow
            demand = demand
            total_loss = 0



            anchor_frame, inflow_tpositive, demand_tpositive, inflow_neg1, demand_neg1, inflow_neg2, demand_neg2 = pairing_sample(speed, inflow, demand , arguments)



            anchor_output, inflow_p,demand_p = ST(speed_input=anchor_frame,inflow_input=inflow_tpositive,demand_input=demand_tpositive)

            _, inflow_n1, _ = ST(inflow_input=inflow_neg1)
            _,_,demand_n1 = ST(demand_input=demand_neg1)
            _, inflow_n2, _= ST(inflow_input=inflow_neg2)
            _, _, demand_n2 = ST(demand_input=demand_neg2)



            
            d_t_positive = tri_distance(anchor_output,inflow_p,demand_p)
            d_t_negative1 = tri_distance(anchor_output,inflow_n1,demand_n1)
            d_t_negative2 = tri_distance(anchor_output,inflow_n2,demand_n2)


            pair_positve = d_t_positive / (d_t_positive + d_t_negative1 + d_t_negative2)
            pair_neg1 = d_t_negative1 / (d_t_negative1 + d_t_negative2 + d_t_positive)
            pair_neg2 = d_t_negative2 / (d_t_negative1 + d_t_negative2 + d_t_positive)


            pairing = arguments.discountfactor_p * torch.clamp(
                arguments.margin_p + pair_positve - pair_neg1 - pair_neg2, min=0).mean()

            total_loss = total_loss + pairing



            speed_anchor,speed_positive,speed_neg,inflow_anchor,inflow_positive,inflow_neg,demand_anchor,demand_positive,demand_neg = temporal_sample(speed, inflow, demand, arguments)


            anchor_tspeed, anchor_tinflow, anchor_tdemand = ST(speed_input=speed_anchor,
                                                  inflow_input= inflow_anchor, demand_input=demand_anchor)

            positive_tspeed, positive_tinflow, positive_tdemand = ST(speed_input=speed_positive,
                                                  inflow_input= inflow_positive, demand_input=demand_positive)

            neg_tspeed, neg_tinflow, neg_tdemand = ST(speed_input=speed_neg,
                                                  inflow_input= inflow_neg, demand_input=demand_neg)


            tpair_positive = distance(anchor_tspeed,positive_tspeed) + distance(anchor_tinflow,positive_tinflow) + distance(anchor_tdemand,positive_tdemand)
            tpair_neg = distance(anchor_tspeed,neg_tspeed) + distance(anchor_tinflow,neg_tinflow) + distance(anchor_tdemand,neg_tdemand)


            temporal = arguments.discountfactor_t * torch.clamp(arguments.margin_t + tpair_positive - tpair_neg,
                                                                min=0).mean()
            total_loss = total_loss + temporal


            speed_anchor, speed_positive, speed_neg, inflow_anchor, inflow_positive, inflow_neg, demand_anchor,demand_positive,demand_neg = spatial_sample(speed, inflow, demand , arguments)


            anchor_tspeed, anchor_tinflow, anchor_tdemand = ST(speed_input=speed_anchor,
                                                   inflow_input=inflow_anchor, demand_input=demand_anchor)

            positive_tspeed, positive_tinflow, positive_tdemand = ST(speed_input=speed_positive,
                                                         inflow_input=inflow_positive, demand_input=demand_positive)

            neg_tspeed, neg_tinflow, neg_tdemand = ST(speed_input=speed_neg,
                                          inflow_input=inflow_neg, demand_input=demand_neg)

            spair_positive = distance(anchor_tspeed,positive_tspeed) + distance(anchor_tinflow,positive_tinflow) + distance(anchor_tdemand,positive_tdemand)
            spair_neg = distance(anchor_tspeed,neg_tspeed) + distance(anchor_tinflow,neg_tinflow) + distance(anchor_tdemand,neg_tdemand)



            spatial = arguments.discountfactor_s * torch.clamp(arguments.margin_s + spair_positive - spair_neg,
                                                               min=0).mean()
            total_loss = total_loss + spatial


            pairing_loss_sum += pairing.item()
            temporal_loss_sum += temporal.item()
            spatial_loss_sum += spatial.item()
            total_loss_sum += total_loss.item()

            num_batches += 1

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

        end_time = time.time()
        dis = end_time - start_time

        average_pairing_loss = pairing_loss_sum / num_batches
        average_temporal_loss = temporal_loss_sum / num_batches
        average_spatial_loss = spatial_loss_sum / num_batches
        average_total_loss = total_loss_sum / num_batches

        logger.info(
            f"")

        learning_rate_scheduler.step()

        if epoch % arguments.save_every == 0 and epoch != 0:
            logger.info('Saving model.')
            save_model(ST, model_filename(arguments.model_name, epoch), arguments.model_folder)


if __name__ == '__main__':
    main()
