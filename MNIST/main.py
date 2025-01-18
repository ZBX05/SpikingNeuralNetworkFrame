import os
import sys

root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from torch import nn
from torch.optim import AdamW
from MNIST.model import LIFNet,SDDTP_LIFNet
from dataset import *
from train import *
import logging

T=100
tau=2
hidden_size=512
lr=1e-3
batch_size=64
epochs=10
# Spiking neuron hyperparameters
surrogate_type='zo'
surrogate_param=0.5
# neuron.surrogate_type='zo'
# # neuron.surrogate_type='arctan'
# # neuron.param=4.0
# neuron.alpha=2.0
# neuron.delta=0.8
criterion=F.mse_loss

assert tau>1,'tau must be greater than 1.0'

# model=LIFNet(T,28*28*1,hidden_size,10,tau,surrogate_type,surrogate_param,criterion)
model=SDDTP_LIFNet(T,28*28*1,hidden_size,10,tau,surrogate_type,surrogate_param,criterion)
# model=WDRTP_LIFNet(T,28*28*1,hidden_size,10,tau,surrogate_type,surrogate_param,train_mode,criterion=criterion)
optimizer=AdamW(model.parameters(),lr=lr)

train_data_loader=get_train_data_loader(batch_size,True)
test_data_loader=get_test_data_loader(batch_size,True)

if __name__=='__main__':
    device=torch.device('cuda:0')
    result_path=os.path.dirname(os.path.abspath(__file__))
    logging.basicConfig(level=logging.INFO,filename=result_path+'./result/train.log',filemode='w')
    logging.info(f'T={T}\ntau={tau}\nhidden_size={hidden_size}\nlr={lr}\nbatch_size={batch_size}\nepochs={epochs}\n\
surrogate_type={surrogate_type}\nsurrogate_param={surrogate_param}\n\n'.strip())
    train(model,train_data_loader,test_data_loader,optimizer,epochs,device,surrogate_param,result_path)