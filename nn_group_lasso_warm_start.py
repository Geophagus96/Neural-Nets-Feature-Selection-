# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:15:20 2022

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
        self.linear1 = torch.nn.Linear(800, 400)
        self.linear2 = torch.nn.Linear(400, 1)
        

    def forward(self, x):
        l1_out = torch.sigmoid(self.linear1(x))
        l2_out = torch.tanh(self.linear2(l1_out))
        return l2_out

def group_lasso_proximal(weights, lr, l):
    neuron_l2 = torch.sum(torch.pow(weights, 2), axis=0)
    penalty_weight = (1-lr*l/torch.sqrt(neuron_l2))*(torch.sqrt(neuron_l2)>(lr*l))
    return weights*penalty_weight

def group_lasso_penalty(weights):
    neuron_l2 = torch.sum(torch.pow(weights, 2), axis=0)
    return torch.sum(torch.sqrt(neuron_l2))

start = time.time()
train_size = 9000
dim_size = 400
valid_size = 2000
beta = torch.cat((torch.rand(dim_size), torch.zeros(dim_size)), dim=0)
#beta = beta.reshape((2*dim_size,1))
x_train = torch.normal(0,1,size=(train_size, 2*dim_size))
x_valid = torch.normal(0,1,size=(valid_size, 2*dim_size))
y_train = torch.mm(x_train, beta.reshape([800,1]))+torch.normal(0,1,size=(train_size,1))
y_valid = torch.mm(x_valid, beta.reshape([800,1]))+torch.normal(0,1,size=(valid_size,1))

device = torch.device('cuda:0')
inputs = Variable(x_train).to(device)
targets = Variable(y_train).to(device)
inputs_valid = Variable(x_valid).to(device)
targets_valid = Variable(y_valid).to(device)

lr = 1e-3

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas =(0.9, 0.999), eps=1e-8)

tol = 1e-3
l = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])

#Single run for neural nets with fixed level of group lasso penalty 

# num_epoch = 40
# for i in range(num_epoch):
#     outputs = model(inputs)
#     optimizer.zero_grad()
#     mse_loss = F.mse_loss(outputs, targets)
#     print(F.mse_loss(model(inputs_valid),targets_valid))
#     loss = mse_loss+(1e-3)*group_lasso_penalty(model.linear1.weight)
#     loss.backward()
#     optimizer.step()
#     #model.linear1.weight = torch.nn.Parameter(group_lasso_proximal(model.linear1.weight, lr, l))

#Neural nets with a sequence level of group lasso penalty, with warm starts
#Warm start could only be applied to a sequence of regularization parameters in an increasing order

c = 0
for j in range(l.size):
    if (j==0):
        old_mse = 1e4
        gap = 0.3
        while(gap>tol):
            outputs = model(inputs)
            optimizer.zero_grad()
            mse_loss = F.mse_loss(outputs, targets)
            loss = mse_loss+l[j]*group_lasso_penalty(model.linear1.weight)
            loss.backward()
            optimizer.step()
            new_mse = F.mse_loss(model(inputs),targets)
            gap = old_mse-new_mse
            old_mse = new_mse
            c+=1
        weights_l2 =  np.sum(np.power(model.linear1.weight.to('cpu').detach().numpy(),2),axis=0)
        init_weight = model.linear1.weight
        print(c)
    else:
        c = 0
        old_mse = 1e4
        outputs = model(inputs)
        model.linear1.weight = init_weight
        optimizer.zero_grad()
        mse_loss = F.mse_loss(outputs, targets)
        loss = mse_loss+l[j]*group_lasso_penalty(model.linear1.weight)
        loss.backward()
        optimizer.step()
        new_mse = F.mse_loss(model(inputs),targets)
        gap = old_mse-new_mse
        old_mse = new_mse
        c+=1
        while(gap>tol):
            outputs = model(inputs)
            optimizer.zero_grad()
            mse_loss = F.mse_loss(outputs, targets)
            loss = mse_loss+l[j]*group_lasso_penalty(model.linear1.weight)
            loss.backward()
            optimizer.step()
            new_mse = F.mse_loss(model(inputs),targets)
            gap = old_mse-new_mse
            old_mse = new_mse
            c+=1
        weights_l2 =  np.vstack((weights_l2, np.sum(np.power(model.linear1.weight.to('cpu').detach().numpy(),2),axis=0)))
        init_weight = model.linear1.weight
        print(c)
        
lag = time.time()-start
print(lag)
print(F.mse_loss(model(inputs_valid),targets_valid))

reg = LinearRegression().fit(np.array(x_train), np.array(y_train))
y_predict = reg.predict(np.array(x_valid))
print(F.mse_loss(torch.tensor(y_predict), y_valid))

