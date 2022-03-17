# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:54:45 2022

@author: Yuze Zhou
"""

import torch 
import time
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from sklearn.linear_model import LinearRegression

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(800, 1)

    def forward(self, x):
        layer1_out = self.linear1(x)
        return layer1_out

def input_decay(weights, eta):
    neuron_l2 = torch.sum(torch.pow(weights, 2), axis=0)
    return torch.sum(neuron_l2/(eta+neuron_l2))


start = time.time()
train_size = 1000
dim_size = 400
valid_size = 500
beta = torch.cat((torch.rand(dim_size), torch.zeros(dim_size)), dim=0)
beta = beta.reshape((2*dim_size,1))
x_train = torch.rand((train_size, 2*dim_size))
x_valid = torch.rand((valid_size, 2*dim_size))
y_train = torch.mm(x_train, beta)+torch.rand((train_size,1))
y_valid = torch.mm(x_valid, beta)+torch.rand((valid_size,1))

device = torch.device('cuda:0')
inputs = Variable(x_train).to(device)
targets = Variable(y_train).to(device)
inputs_valid = Variable(x_valid).to(device)
targets_valid = Variable(y_valid).to(device)

model = MLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)

phi, eta = 0.1, 0.1

num_epoch = 21000
for i in range(num_epoch):
    outputs = model(inputs)
    optimizer.zero_grad()
    input_regularizer = input_decay(model.linear1.weight, eta) 
    mse_loss = F.mse_loss(outputs, targets)
    loss =  mse_loss + phi*input_regularizer 
    loss.backward() 
    optimizer.step()
    
lag = time.time()-start
print(lag)
print(F.mse_loss(model(inputs_valid),targets_valid))

reg = LinearRegression().fit(np.array(x_train), np.array(y_train))
y_predict = reg.predict(x_valid)
print(F.mse_loss(torch.tensor(y_predict), y_valid))

