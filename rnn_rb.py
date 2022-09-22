# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:55:59 2022

@author: geoph
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

class tsDataset(torch.utils.data.Dataset):
    
    
    def __init__(self, file_name, nlag, nsample):
        self.ts = file_name
        self.lag = nlag
        self.keys = file_name['ukey'].unique()
        self.nsamp = nsample

    #Here we don't follow the routine __get
    def __getitem__(self, idx):
        keystate = False
        while (keystate==False):
            ukey = np.random.choice(self.keys)
            df_ukey = self.ts.loc[self.ts['ukey']==ukey,]
            max_date = len(df_ukey)
            if (max_date<=self.lag):
                keystate = False
            else:
                c_date = np.random.randint(0, max_date-self.lag)
                if (df_ukey['date'].iloc[(c_date+self.lag-1)] == (c_date+self.lag-1)):
                    keystate = True
                else:
                    keystate = False
        seq_xtrain = df_ukey.iloc[c_date:(c_date+self.lag), 2:802]
        seq_ytrain = df_ukey['y'].iloc[(c_date+self.lag-1)]
        seq_xtrain = torch.tensor(np.array(seq_xtrain))
        seq_ytrain = torch.tensor(np.array(seq_ytrain))
        
        return seq_xtrain, seq_ytrain
    
    def __len__(self):
        return self.nsamp


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
    
    def __init__(self, lr, tol, batchsize, n_hidden, length, n_train, n_valid):
        self.lr = lr 
        self.tol = tol
        self.batchsize = batchsize
        self.n_hidden = n_hidden
        self.n_len = length
        self.n_train = n_train
        self.n_valid = n_valid
    
    def train(self, train, valid, device, max_iter):
        self.n_feat = train.shape[1]-4
        trainset = tsDataset(train, self.n_len, self.n_train)
        trainloader = DataLoader(trainset, batch_size=self.batchsize, shuffle=True)
        validset = tsDataset(valid, self.n_len, self.n_valid)
        validloader = DataLoader(validset, batch_size=self.batchsize, shuffle=True)
        model = multi_to_one_rnn(self.n_feat, self.n_hidden).to(device)
        c = 0 
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0)
        old_cor = -0.1
        gap = 0
        y_train = np.zeros(self.n_train)
        train_output = np.zeros(y_train.size)
        y_valid = np.zeros(self.n_valid)
        valid_output = np.zeros(y_valid.size)
        while ((gap>=self.tol) and (c<=max_iter)):
            for _, (x_batch, y_batch) in enumerate(trainloader):
                inputs = Variable(x_batch).to(device)
                targets = Variable(y_batch).to(device)
                outputs = model(inputs.float()).to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(outputs.float(), targets.float())
                loss.backward()
                optimizer.step()
            c += 1
            
            for id_batch, (x_batch, y_batch) in enumerate(validloader):
                inputs = Variable(x_batch).to(device)
                outputs = model(inputs.float())
                outputs = outputs.to('cpu').detach().numpy()
                if((y_batch.size()[0]) == self.batchsize):
                    y_valid[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = y_batch.detach().numpy()
                    valid_output[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = outputs
                else:
                    y_valid[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = y_batch.detach().numpy()
                    valid_output[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = outputs
            new_cor = np.corrcoef(y_valid, valid_output)[0,1]
            gap = new_cor-old_cor
            old_cor = new_cor
        
        print('total number of epoch:', c)
        for id_batch, (x_batch, y_batch) in enumerate(trainloader):
            inputs = Variable(x_batch).to(device)
            outputs = model(inputs.float())
            outputs = outputs.to('cpu').detach().numpy()
            if((y_batch.size()[0]) == self.batchsize):
                train_output[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = outputs
            else:
                train_output[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = outputs
        train_cor = np.corrcoef(y_train, train_output)[0,1]
        
        self.model = model
        
        return train_cor, new_cor, train_output, valid_output
    
    
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


train_size = 9000
dim_size = 400
valid_size = 2000
ukey_train = np.repeat(np.arange(180), 50)
ukey_valid = np.repeat(np.arange(40), 50)
date_train = np.tile(np.arange(50), 180)
date_valid = np.tile(np.arange(50), 40)
flag_train = np.random.binomial(size=9000, n=1, p = 0.995)
flag_valid = np.random.binomial(size=2000, n=1, p = 0.995)
beta = torch.cat((torch.rand(dim_size), torch.zeros(dim_size)), dim=0)
x_train = torch.normal(0,1,size=(train_size, 2*dim_size))
x_valid = torch.normal(0,1,size=(valid_size, 2*dim_size))
y_train = torch.mm(x_train, beta.reshape([800,1]))+torch.normal(0,1,size=(train_size,1))
y_valid = torch.mm(x_valid, beta.reshape([800,1]))+torch.normal(0,1,size=(valid_size,1))
x_train = x_train.detach().numpy()
y_train = y_train.detach().numpy()
y_train.resize([9000])
x_valid = x_valid.detach().numpy()
y_valid = y_valid.detach().numpy()
y_valid.resize([2000])
df_train = pd.concat([pd.Series(ukey_train, name='ukey'), pd.Series(date_train, name='date'), pd.DataFrame(x_train), pd.Series(y_train, name='y'), pd.Series(flag_train, name='flag')], axis=1)
df_valid = pd.concat([pd.Series(ukey_valid, name='ukey'), pd.Series(date_valid, name='date'), pd.DataFrame(x_valid), pd.Series(y_valid, name='y'), pd.Series(flag_valid, name='flag')], axis=1)
df_train = df_train.loc[df_train['flag']==1,]
df_valid = df_valid.loc[df_valid['flag']==1,]

lr = 5e-4
rnn_model = vanilla_rnn(lr, -3e-4, 20, 100, 10, 7200, 1600)
   

train_cor, valid_cor, train_output, valid_output= rnn_model.train(df_train, df_valid, 'cuda:0', 10)