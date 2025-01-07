from torch import nn
from torch.optim import AdamW
import neuron
from neuron import LIFNode
from dataset import *
from train import *

T=100
tau=2
hidden_size=512
lr=1e-3
batch_size=64
epochs=10
#!目前未使用整合至LIFNode内的ActFun，因此这组参数实际上并未使用#
surrogate_type='arctan'
surrogate_param=2.0
#!###########################################################
neuron.surrogate_type='arctan'
neuron.param=2.0

assert tau>1,'tau must be greater than 1.0'

class LIF_Net(nn.Module):
    def __init__(self,T:int,input_size:int,hidden_size:int,output_size:int,tau:float,surrogate_type:str,surrogate_param:float) -> None:
        super(LIF_Net,self).__init__()
        self.T=T
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=nn.Linear(input_size,hidden_size)
        self.lif_1=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        self.fc_2=nn.Linear(hidden_size,output_size)
        self.lif_2=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        x=self.flatten(input)
        # print(x.shape)
        # exit()
        x=self.fc_1(x)
        x=self.lif_1(x)
        x=self.fc_2(self.flatten(x))
        x=self.lif_2(x)
        return x.mean(1)

# model=ANN_Net(28*28,hidden_size,10)
model=LIF_Net(T,28*28*1,hidden_size,10,tau,surrogate_type,surrogate_param)
optimizer=AdamW(model.parameters(),lr=lr)

train_data_loader=get_train_data_loader(batch_size,True)
test_data_loader=get_test_data_loader(batch_size,True)

if __name__=='__main__':
    device=torch.device('cuda:0')
    train(model,train_data_loader,test_data_loader,optimizer,epochs,device)