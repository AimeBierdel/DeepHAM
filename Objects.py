
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.data import Dataset

print("Packages loaded")


device = torch.device('cpu')

#### Define the class for our neural network. Give it a function forward for the iteration of the network
#### We use ReLU activations but give the possibility of a sigmoid output at the end to replicate the paper
#### The optional argument is GMparam. If it is not None, then we create a generalized moment.
#### GMparams = [input_size, hidden_size, nlayers, output_size, lastActivFunc] like the regular parameters



class GMomentNet(nn.Module):
    def __init__(self, input_size,hidden_size,nlayers, output_size,lastActivFunc=None):
        super(GMomentNet,self).__init__()
        self.firstl = nn.Linear(input_size,hidden_size)
        nn.init.normal_(self.firstl.weight, mean=1/input_size, std=1.0)
        if nlayers > 1 :
            self.hiddenl = nn.Linear(hidden_size,hidden_size)
            nn.init.normal_(self.hiddenl.weight, mean=1/hidden_size, std=1.0)
        self.lastl = nn.Linear(hidden_size,output_size)
        nn.init.normal_(self.lastl.weight, mean=1/hidden_size, std=1.0)
        self.nlayers = nlayers
        self.lastActivFunc = lastActivFunc
        self.inputsize= torch.tensor([input_size*1.])
        self.hiddensize = torch.tensor([hidden_size*1.])
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.sigmoid = nn.Sigmoid()
     
    def forward(self,x):
        ## Compute the Generalized moment if parameters have been supplied to the builder
        
        out = self.firstl(x)   ### Apply the policy/value function fitting neural network
        #out = torch.div(out, self.hiddensize)
        out = self.relu(out)
        if self.nlayers > 1:
            for i in range(0,self.nlayers):
                out = self.hiddenl(out)
                #out = torch.div(out, self.hiddensize)
                out = self.relu(out)
        out = self.lastl(out)
        #out = torch.div(out, self.hiddensize)
        if self.lastActivFunc == "sigmoid" :
            out = self.sigmoid(out)
        else : 
            out = self.relu(out)
        return out



class FittingNet(nn.Module):
    def __init__(self, input_size,hidden_size,nlayers, output_size,lastActivFunc=None):
        super(FittingNet,self).__init__()
        self.firstl = nn.Linear(input_size,hidden_size,bias=True)
        nn.init.normal_(self.firstl.weight, mean=0., std=1.0)
        if nlayers > 1 :
            self.hiddenl = nn.Linear(hidden_size,hidden_size)
            nn.init.normal_(self.hiddenl.weight, mean=0., std=1.0)
        self.lastl = nn.Linear(hidden_size,output_size)
        nn.init.normal_(self.lastl.weight, mean=0., std=1.0)
        self.nlayers = nlayers
        self.lastActivFunc = lastActivFunc
        self.inputsize= torch.tensor([input_size*1.])
        self.hiddensize = torch.tensor([hidden_size*1.])
        self.relu = nn.LeakyReLU(negative_slope=0.001)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.norm = input_size*hidden_size* nlayers *output_size
    def forward(self,x):
        ## Compute the Generalized moment if parameters have been supplied to the builder
        
        out = self.firstl(x)   ### Apply the policy/value function fitting neural network
        out = self.relu(out)
        if self.nlayers > 1:
            for i in range(0,self.nlayers):
                out = self.hiddenl(out)
                out = self.relu(out)
        out = self.lastl(out)
        out = torch.div(out, self.hiddensize)
        if self.lastActivFunc == "sigmoid" :
            out = out.clamp(min = -torch.sqrt(torch.tensor([3])), max = torch.sqrt(torch.tensor([3])))
            out = self.sigmoid(out)
        elif self.lastActivFunc == "softmax":
            out = self.softmax(out)
        else : 
            out = self.relu(out)
        return out

class ValueModel(nn.Module):
    def __init__(self,Q,Valmodel,macroEngine):
        super(ValueModel,self).__init__()
        self.val = Valmodel 
        self.Q = Q 
        self.macroEngine = macroEngine

    def forward(self,x):
        K = x[:,2:3]
        out = self.macroEngine.GMoment(K,self.Q)
        out = out.expand(x.size(0),1)  
        Input = torch.cat((x,out),1)
        out = self.val(Input) 
        return out

class PolicyModel(nn.Module):
    def __init__(self,Q,Polmodel,macroEngine):
        super(PolicyModel,self).__init__()
        self.pol = Polmodel 
        self.Q = Q 
        self.macroEngine = macroEngine

    def forward(self,x):
        out = self.macroEngine.GMoment(x[:,2].view((x.size(0),1)),self.Q)
        out = out.expand(x.size(0),1)   
        Input = torch.cat((x,out),1)
        out = self.pol(Input) 
        return out
    
    def forwardIndex(self,x,index):
        # This time we only compute gradient for the computation of the policy of agent index 
        out = self.macroEngine.GMoment(x[:,2].view((x.size(0),1)),self.Q)
        out = out.expand(x.size(0),1) 
        Input = torch.cat((x,out),1)
        with torch.no_grad():
            out = self.pol(Input) 
        out[index,:] = self.pol(Input[index,:])
        return out

class StatDataset(Dataset):
    def __init__(self,X):
        self.X = X
        self.nEconomies = X.size(0)
        self.nAgents = X.size(1)
        self.nFeatures = X.size(2)

    def __getitem__(self,index):
        return self.X[index]
    
    def __len__(self):
        return self.nEconomies
    
    def update(self,X):
        self.X = X
        self.nEconomies = X.size(0)
        self.nAgents = X.size(1)
        self.nFeatures = X.size(2)

