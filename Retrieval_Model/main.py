import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader

from Retrieval_Model.model.transposeMLP import Net
from Retrieval_Model.test import test_stage
from Retrieval_Model.train import train_stage
from Retrieval_Model.utils import WarmupMultiStepLR

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load feature
from Retrieval_Model.utils import split_dataset, ImageDataSet, RandomIdentitySampler

feature = pd.read_csv("../data/features/DCTHistfeats.csv", header=None, names=[i for i in range(12288)])
Y_train = np.zeros((10000, 1), dtype=int)
j = 0
for i in range(0, 10000, 100):
    Y_train[i:i + 100] = j
    j = j + 1

dataset = np.concatenate((feature, Y_train), axis=1)
dataset = pd.DataFrame(dataset)

batchsize = 100
num_class = 10
Xp_train, yp_train, Xp_test, yp_test = split_dataset(dataset)
train_data = ImageDataSet(Xp_train, yp_train.tolist())
test_data = ImageDataSet(Xp_test, yp_test)
train_loader = DataLoader(dataset=train_data, batch_size=batchsize,
                          sampler=RandomIdentitySampler(train_data, batchsize, num_class))
valid_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)

# load model
model = Net(seq_len=64,
            d_model=192,
            token_dim=768,
            channel_dim=256,
            represent_dim=128,
            n_blocks=12,
            n_classes=100, )

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=[60, 70])

# train model
train_stage(model, train_loader, optimizer, scheduler)

# valid model
test_stage(model, Xp_test, yp_test)
