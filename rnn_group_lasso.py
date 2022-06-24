# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:02:41 2022

@author: geoph
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt

def build_time_series_data(x, y, TIME_STEPS):
    row_dim = TIME_STEPS
    col_dim = x.shape[1]
    third_dim = x.shape[0] - TIME_STEPS
    x_time_series_data = torch.zeros((third_dim, row_dim, col_dim))
    y_time_series_data = torch.zeros((third_dim))
    for i in range(third_dim):
        x_time_series_data[i] = x[i:TIME_STEPS+i]
        y_time_series_data[i] = y[TIME_STEPS+i-1]
    return x_time_series_data, y_time_series_data

def group_lasso_penalty(weights):
    neuron_l2 = torch.sum(torch.pow(weights,2), axis=0)
    return torch.sum(torch.sqrt(neuron_l2))


class multi_to_one_rnn(torch.nn.Module):
    
    def __init__(self, n_input, n_hidden):
        super(multi_to_one_rnn, self).__init__()
        self.Rnn = torch.nn.RNN(input_size=n_input, hidden_size=n_hidden, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        self.n_length = x.size()[1]
        n_batch = x.size()[0]
        y_pred = torch.zeros(n_batch)
        _ , out_last = self.Rnn(x)
        for j in range(n_batch):
            h_batch = out_last[0,j,:] 
            y_pred[j] = self.linear(torch.sigmoid(h_batch))
        return y_pred

class vanilla_rnn():
    
    def __init__(self, l, lr, tol, batchsize, n_hidden):
        self.l = l
        self.lr = lr 
        self.tol = tol
        self.batchsize = batchsize
        self.n_hidden = n_hidden
    
    def train(self, x_train, y_train, x_valid, y_valid, device, max_iter):
        self.n_feat = x_train.size()[2]
        # x_train = torch.tensor(x_train)
        # y_train = torch.tensor(y_train)
        # x_valid = torch.tensor(x_valid)
        # y_valid = torch.tensor(y_valid)
        trainset = TensorDataset(x_train, y_train)
        validset = TensorDataset(x_valid, y_valid)
        trainloader = DataLoader(trainset, batch_size=self.batchsize, shuffle=False)
        validloader = DataLoader(validset, batch_size=self.batchsize, shuffle=False)
        model = multi_to_one_rnn(self.n_feat, self.n_hidden).to(device)
        
        train_cor = np.zeros(self.l.size)
        valid_cor = np.zeros(self.l.size)
        weights_hh_path = torch.zeros([self.l.size, self.n_hidden, self.n_hidden])
        bias_hh_path = torch.zeros([self.l.size, self.n_hidden])
        weights_ih_path = torch.zeros([self.l.size, self.n_hidden, self.n_feat])
        bias_ih_path = torch.zeros([self.l.size, self.n_hidden])
        weights_linear_path = torch.zeros([self.l.size, self.n_hidden])
        bias_linear_path = torch.zeros([self.l.size])
        
        input_weights_path = np.zeros([self.l.size, self.n_feat])
        
        for j in range(self.l.size):
            c = 0
            old_cor = -0.1
            gap = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr[j], betas=(0.9, 0.999), eps=1e-6, weight_decay=0)
            train_output = np.zeros(y_train.size()[0])
            valid_output = np.zeros(y_valid.size()[0])
            while ((gap>=self.tol) and (c<=max_iter)):
                for _, (x_batch, y_batch) in enumerate(trainloader):
                    inputs = Variable(x_batch).to(device)
                    targets = Variable(y_batch).to(device)
                    outputs = model(inputs.float()).to(device)
                    optimizer.zero_grad()
                    loss = F.mse_loss(outputs.float(), targets.float())+self.l[j]*group_lasso_penalty(model.Rnn.weight_ih_l0)
                    loss.backward()
                    optimizer.step()
                c += 1
        
                for id_batch, (x_batch, y_batch) in enumerate(validloader):
                    inputs = Variable(x_batch).to(device)
                    outputs = model(inputs.float())
                    outputs = outputs.to('cpu').detach().numpy()
                    if((y_batch.size()[0]) == self.batchsize):
                        valid_output[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = outputs
                    else:
                        valid_output[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = outputs
                y_valid_np = y_valid.detach().numpy()
                new_cor = np.corrcoef(y_valid_np, valid_output)[0,1]
                gap = new_cor-old_cor
                old_cor = new_cor
            
            valid_cor[j] = new_cor
            print('total number of epoch:', c)
            for id_batch, (x_batch, y_batch) in enumerate(trainloader):
                inputs = Variable(x_batch).to(device)
                outputs = model(inputs.float())
                outputs = outputs.to('cpu').detach().numpy()
                if((y_batch.size()[0]) == self.batchsize):
                    train_output[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = outputs
                else:
                    train_output[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = outputs
            y_train_np = y_train.detach().numpy()
            train_cor[j] = np.corrcoef(y_train_np, train_output)[0,1]
            
            weights_hh_path[j,:,:] = model.Rnn.weight_hh_l0
            weights_ih_path[j,:,:] = model.Rnn.weight_ih_l0
            weights_linear_path[j,:] = model.linear.weight
            bias_hh_path[j,:] = model.Rnn.bias_hh_l0
            bias_ih_path[j,:] = model.Rnn.bias_ih_l0
            bias_linear_path[j] = model.linear.bias
            
            input_weights_path[j, ] = np.sum(np.power(model.Rnn.weight_ih_l0.to('cpu').detach().numpy(),2),axis=0)
        
        best_l = np.argmax(valid_cor)
        model.Rnn.weight_hh_l0 = torch.nn.Parameter(weights_hh_path[best_l,:,:])
        model.Rnn.weight_ih_l0 = torch.nn.Parameter(weights_ih_path[best_l,:,:])
        model.linear.weight = torch.nn.Parameter(weights_linear_path[best_l,:])
        model.Rnn.bias_hh_l0 = torch.nn.Parameter(bias_hh_path[best_l,:])
        model.Rnn.bias_ih_l0 = torch.nn.Parameter(bias_ih_path[best_l,:])
        model.linear.bias = torch.nn.Parameter(bias_linear_path[best_l])
        input_weights_path[j, ] = np.sum(np.power(model.Rnn.weight_ih_l0.to('cpu').detach().numpy(),2),axis=0)
        
        self.model = model
        return train_cor, valid_cor, train_output, valid_output, input_weights_path
    
    
    def test(self, x_test, device):
        x_test =  torch.Tensor(x_test)
        testloader = DataLoader(x_test, batch_size=self.batchsize, shuffle=False)
        test_output = np.zeros(x_test.size()[0])
        for id_batch, (x_batch, y_batch) in enumerate(testloader):
            inputs = Variable(x_batch).to(device)
            outputs = self.model(inputs.float())
            outputs = outputs.to('cpu').detach().numpy()
            if((y_batch.size()[0]) == self.batchsize):
                test_output[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = outputs
            else:
                test_output[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = outputs
        return test_output

if __name__ == '__main__':
    train_size = 9000
    dim_size = 400
    valid_size = 2000
    beta = torch.cat((torch.rand(dim_size), torch.zeros(dim_size)), dim=0)
    x_train = torch.normal(0,1,size=(train_size, 2*dim_size))
    x_valid = torch.normal(0,1,size=(valid_size, 2*dim_size))
    y_train = torch.mm(x_train, beta.reshape([800,1]))+torch.normal(0,1,size=(train_size,1))
    y_valid = torch.mm(x_valid, beta.reshape([800,1]))+torch.normal(0,1,size=(valid_size,1))
    l = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    lr = np.array([5e-4, 5e-4, 5e-4,5e-4, 5e-4])
    rnn_model = vanilla_rnn(l, lr, -2e-4, 20, 200)
    x_time_series_train, y_time_series_train = build_time_series_data(x_train, y_train, 10)
    x_time_series_valid, y_time_series_valid = build_time_series_data(x_valid, y_valid, 10)

    train_cor, new_cor, train_output, valid_output, path = rnn_model.train(x_time_series_train, y_time_series_train, x_time_series_valid, y_time_series_valid, 'cuda:0', 10)
        
    for j in range(l.size):
        fig = plt.figure();
        signal_weights = path[j, 1:400]
        noise_weights = path[j, 400:-1]
        plt.hist(signal_weights, weights=np.ones_like(signal_weights)/len(signal_weights), label='signal');
        plt.hist(noise_weights, weights=np.ones_like(noise_weights)/len(noise_weights), label='noise');
        plt.legend();
        
        