import torch 
import time
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

train_size = 300000
dim_size = 400
valid_size = 100000
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

batch_size = 250
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_valid, y_valid)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out
 
def group_lasso_penalty(weights):
    neuron_l2 = torch.sum(torch.pow(weights, 2), axis=0)
    return torch.sum(torch.sqrt(neuron_l2))
 
input_dim = x_train.size(1)
hidden_dim = 200
layer_dim = 1
output_dim = 1 

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
error = nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 15
l = 0.1

train_pred = np.zeros(train_size)
valid_pred = np.zeros(valid_size)
for epoch in range(num_epochs):
    for i, (binput, btarget) in enumerate(train_loader):

        train  = Variable((binput.view(-1, 1, input_dim)))
        labels = Variable(btarget)
            
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        #MSE loss with regularization
        loss = error(outputs, labels)+l*group_lasso_penalty(model.rnn.weight_ih_l0)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
    for i, (binput, btarget) in enumerate(train_loader):
        train  = Variable((binput.view(-1, 1, input_dim)))
        outputs = model(train)
        train_pred[(i*batch_size):((i+1)*batch_size)] = np.resize(outputs.detach().numpy(), batch_size)
    for i, (binput, btarget) in enumerate(test_loader):
        train  = Variable((binput.view(-1, 1, input_dim)))
        outputs = model(train)
        valid_pred[(i*batch_size):((i+1)*batch_size)] = np.resize(outputs.detach().numpy(), batch_size)
    print(np.corrcoef(train_pred, np.resize(y_train.detach().numpy(), train_size))[0][1])
    print(np.corrcoef(valid_pred, np.resize(y_valid.detach().numpy(), valid_size))[0][1])

import matplotlib.pyplot as plt
norms = np.sum(np.power(model.rnn.weight_ih_l0.detach().numpy(),2), axis=0)
fig = plt.figure()
plt.hist(norms[0:400], edgecolor='black', weights=np.ones_like(norms[0:400]) / len(norms[0:400]))
plt.hist(norms[400:-1], edgecolor='red', weights=np.ones_like(norms[400:-1]) / len(norms[400:-1]))

