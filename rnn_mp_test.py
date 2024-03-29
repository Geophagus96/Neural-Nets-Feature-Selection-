# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 11:22:38 2022

@author: geoph
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool
import logging 



class tsDataset(torch.utils.data.Dataset):
    
    
    def __init__(self, file_name, nlag, nsample, seed):
        self.ts = file_name
        self.lag = nlag
        self.keys = file_name['ukey'].unique()
        self.nsamp = nsample
        np.random.seed(seed)

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
    
    def __init__(self, n_input, n_hidden, n_neuron):
        super(multi_to_one_rnn, self).__init__()
        self.norm = torch.nn.BatchNorm1d(n_input)
        self.Rnn = torch.nn.RNN(input_size=n_input, hidden_size=n_hidden, num_layers=2, batch_first=True)
        self.linear1 = torch.nn.Linear(n_hidden, n_neuron)
        self.linear2 = torch.nn.Linear(n_neuron, 1)
        self.n_hidden = n_hidden
        self.n_neuron = n_neuron
        self.n_feat = n_input
    
    def forward(self, x, h0):
        self.n_length = x.size()[1]
        n_batch = x.size()[0]
        y_pred = torch.zeros(n_batch)
        x = self.norm(x.permute([0,2,1]))
        interim , out_last = self.Rnn(x.permute([0,2,1]), h0)
        for j in range(n_batch):
            h_batch = out_last[0,j,:] 
            h_output = self.linear1(torch.sigmoid(h_batch))
            y_pred[j] = self.linear2(torch.sigmoid(h_output))
        return y_pred

class vanilla_rnn():
    
    def __init__(self, l, lr, tol, batchsize, n_hidden, n_neuron, length, n_train, n_valid):
        self.l = l
        self.lr = lr 
        self.tol = tol
        self.batchsize = batchsize
        self.n_hidden = n_hidden
        self.n_len = length
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_neuron = n_neuron
        
    
    def train(self, train, valid, device, max_iter):
        self.n_feat = train.shape[1]-4
        trainset = tsDataset(train, self.n_len, self.n_train, None)
        trainloader = DataLoader(trainset, batch_size=self.batchsize, shuffle=True)
        validset = tsDataset(valid, self.n_len, self.n_valid, 10)
        validloader = DataLoader(validset, batch_size=self.batchsize, shuffle=True)
        
        model = multi_to_one_rnn(self.n_feat, self.n_hidden, self.n_neuron).to(device)
        train_cor = np.zeros(self.l.size)
        valid_cor = np.zeros(self.l.size)
        weights_hh_path_0 = torch.zeros([self.l.size, self.n_hidden, self.n_hidden])
        bias_hh_path_0 = torch.zeros([self.l.size, self.n_hidden])
        weights_hh_path_1 = torch.zeros([self.l.size, self.n_hidden, self.n_hidden])
        bias_hh_path_1 = torch.zeros([self.l.size, self.n_hidden])
        weights_ih_path = torch.zeros([self.l.size, self.n_hidden, self.n_feat])
        bias_ih_path = torch.zeros([self.l.size, self.n_hidden])
        weights_linear0_path = torch.zeros([self.l.size, self.n_hidden, self.n_neuron])
        weights_linear1_path = torch.zeros([self.l.size, self.n_neuron])
        bias_linear0_path = torch.zeros([self.l.size, self.n_neuron])
        bias_linear1_path = torch.zeros([self.l.size])
        
        input_weights_path = np.zeros([self.l.size, self.n_feat])
        print('model and dataloader set up')
        for j in range(self.l.size):
            c = 0
            old_cor = -0.1
            gap = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr[j], betas=(0.9, 0.999), eps=1e-6, weight_decay=0)
           
            
            y_train = np.zeros(self.n_train)
            
            train_output = np.zeros(y_train.size)
            y_valid = np.zeros(self.n_valid)
            valid_output = np.zeros(y_valid.size)
            h0 = torch.zeros([2, self.batchsize, self.n_hidden])
            while ((gap>=self.tol) and (c<=max_iter)):
                
                for _, (x_batch, y_batch) in enumerate(trainloader):
                    inputs = Variable(x_batch).to(device)
                    targets = Variable(y_batch).to(device)
                    outputs = model(inputs.float(), h0.to(device)).to(device)
                    optimizer.zero_grad()
                    loss = F.mse_loss(outputs.float(), targets.float())+self.l[j]*group_lasso_penalty(model.Rnn.weight_ih_l0)
                    loss.backward()
                    optimizer.step()
                c += 1
                for id_batch, (x_batch, y_batch) in enumerate(validloader):
                    inputs = Variable(x_batch).to(device)
                    outputs = model(inputs.float(), h0.to(device))
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
            valid_cor[j] = new_cor
            print('total number of epoch:', c)
            for id_batch, (x_batch, y_batch) in enumerate(trainloader):
                inputs = Variable(x_batch).to(device)
                outputs = model(inputs.float(),h0.to(device))
                outputs = outputs.to('cpu').detach().numpy()
                if((y_batch.size()[0]) == self.batchsize):
                    y_train[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = y_batch.detach().numpy()
                    train_output[(id_batch*self.batchsize):((id_batch+1)*self.batchsize)] = outputs
                else:
                    y_train[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = y_batch.detach().numpy()
                    train_output[(id_batch*self.batchsize):(id_batch*self.batchsize+y_batch.size()[0])] = outputs
            train_cor[j] = np.corrcoef(y_train, train_output)[0,1]
            
            weights_hh_path_0[j,:,:] = model.Rnn.weight_hh_l0
            weights_hh_path_1[j,:,:] = model.Rnn.weight_hh_l1
            weights_ih_path[j,:,:] = model.Rnn.weight_ih_l0
            weights_linear0_path[j,:,:] = model.linear1.weight
            weights_linear1_path[j,:] = model.linear2.weight
            bias_hh_path_0[j,:] = model.Rnn.bias_hh_l0
            bias_hh_path_1[j,:] = model.Rnn.bias_hh_l1
            bias_ih_path[j,:] = model.Rnn.bias_ih_l0
            bias_linear0_path[j,:] = model.linear1.bias
            bias_linear1_path[j] = model.linear2.bias
            
            input_weights_path[j, ] = np.sum(np.power(model.Rnn.weight_ih_l0.to('cpu').detach().numpy(),2),axis=0)
        
        best_l = np.argmax(valid_cor)
        model.Rnn.weight_hh_l0 = torch.nn.Parameter(weights_hh_path_0[best_l,:,:])
        model.Rnn.weight_hh_l1 = torch.nn.Parameter(weights_hh_path_1[best_l,:,:])
        
        model.Rnn.weight_ih_l0 = torch.nn.Parameter(weights_ih_path[best_l,:,:])
        model.linear1.weight = torch.nn.Parameter(weights_linear0_path[best_l,:,:])
        model.linear2.weight = torch.nn.Parameter(weights_linear1_path[best_l,:])
        model.Rnn.bias_hh_l0 = torch.nn.Parameter(bias_hh_path_0[best_l,:])
        model.Rnn.bias_hh_l1 = torch.nn.Parameter(bias_hh_path_1[best_l,:])
        model.Rnn.bias_ih_l0 = torch.nn.Parameter(bias_ih_path[best_l,:])
        model.linear1.bias = torch.nn.Parameter(bias_linear0_path[best_l,:])
        model.linear2.bias = torch.nn.Parameter(bias_linear1_path[best_l])
        
        self.model = model
        return train_cor, valid_cor, train_output, valid_output, input_weights_path
    
    
    def test(self, x_test, device):
        x_test =  torch.Tensor(x_test)
        testloader = DataLoader(x_test, batch_size=self.batchsize, shuffle=False)
        test_output = np.zeros(x_test.size()[0])
        h0 = torch.zeros([1, self.batchsize, self.n_hidden])

        for id_batch, (x_batch, y_batch) in enumerate(testloader):
            inputs = Variable(x_batch).to(device)
            outputs = self.model(inputs.float(), h0.to(device))
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

l = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0])
lr = np.array([5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4])
#rnn_model = vanilla_rnn(l, lr, -3e-4, 20, 100, 100, 10, 7200, 1600)
#train_cor, valid_cor, train_output, valid_output, path = rnn_model.train(df_train, df_valid, 'cuda:0', 10)
 
def rnn_output(input_dict):
    lr = input_dict['lr']
    rnn_model = vanilla_rnn(l, lr, -3e-4, 20, 100, 100, 10, 7200, 1600)
    train_cor, valid_cor, train_output, valid_output, path = rnn_model.train(df_train, df_valid, 'cuda:0', 10)
    with open(input_dict['log'], 'w') as f:
        f.write('learning rates:')
        np.savetxt(f, lr.reshape([1,lr.size]), fmt = '%.8f')
        f.write('training correlation')
        np.savetxt(f, train_cor.reshape([1,train_cor.size]), fmt = '%.8f')
        f.write('validation correlation')
        np.savetxt(f, valid_cor.reshape([1,valid_cor.size]), fmt = '%.8f')

if __name__ == '__main__':
    dict_1 = {'lr':np.array([5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4]), 'log':'file1.txt'}
    dict_2 = {'lr':np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]), 'log':'file2.txt'}
    dicts = np.array([dict_1, dict_2])
    num_threads = 4
    with Pool(num_threads) as pool: 
        results = pool.map(rnn_output, dicts)

# for j in range(l.size):
#     fig = plt.figure();
#     signal_weights = path[j, 1:400]
#     noise_weights = path[j, 400:-1]
#     plt.hist(signal_weights, weights=np.ones_like(signal_weights)/len(signal_weights), label='signal');
#     plt.hist(noise_weights, weights=np.ones_like(noise_weights)/len(noise_weights), label='noise');
#     plt.legend();
    

