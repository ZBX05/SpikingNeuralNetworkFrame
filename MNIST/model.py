import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron import LIFNode
from module import FCBlock,FixedFCBlock,ConnectedFCBlock,TestBlock
from typing import Any

class LIFNet(nn.Module):
    def __init__(self,T:int,input_size:int,hidden_size:int,output_size:int,tau:float,surrogate_type:str='zo',
                 surrogate_param:float=0.5,criterion:Any=F.mse_loss) -> None:
        super(LIFNet,self).__init__()
        self.T=T
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=nn.Linear(input_size,hidden_size)
        self.lif_1=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        self.fc_2=nn.Linear(hidden_size,output_size)
        self.lif_2=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        self.criterion=criterion
    
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor|torch.Tensor]:
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        x=self.flatten(input)
        x=self.fc_1(x)
        x=self.lif_1(x)
        x=self.fc_2(self.flatten(x))
        x=self.lif_2(x)
        return x.mean(1),self.criterion(x.mean(1),labels)

class SDDTP_LIFNet(nn.Module):
    def __init__(self,T:int,input_size:int,hidden_size:int,output_size:int,tau:float,surrogate_type:str='zo',surrogate_param:float=0.5,
                 criterion:Any=F.mse_loss) -> None:
        super(SDDTP_LIFNet,self).__init__()
        self.T=T
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=FCBlock(input_size,hidden_size,output_size,tau,surrogate_type,surrogate_param)
        self.fc_2=FCBlock(hidden_size,output_size,output_size,tau,surrogate_type,surrogate_param)
        self.criterion=criterion

    def step(self,input:torch.Tensor,labels:torch.Tensor,training:bool=True) -> tuple[torch.Tensor,torch.Tensor]:
        x=self.flatten(input)
        if training:
            loss_sum=0

            x,y_t=self.fc_1(x,True)
            loss_sum+=self.criterion(y_t.mean(1),labels)
            x=x.clone().detach()

            x,y_t=self.fc_2(self.flatten(x))
            loss_sum+=self.criterion(x.mean(1),labels)
            return x,loss_sum
        else:
            x,_=self.fc_1(x)
            x,_=self.fc_2(self.flatten(x))
            return x,self.criterion(x.mean(1),labels)
        
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        self.reset()
        # if self.training:
        #     input=input.unsqueeze(1).repeat(1,+self.burnin+self.T,1,1,1)
        #     with torch.no_grad():
        #         self.step(input[:,:self.burnin,...],labels,True)
        #     x,loss_sum=self.step(input[:,self.burnin:,...],labels,True)
        if self.training:
            input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
            x,loss_sum=self.step(input,labels,True)
        else:
            input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
            x,loss_sum=self.step(input,labels,False)
        return x.mean(1),loss_sum
    
    def reset(self) -> None:
        self.fc_1.reset()
        self.fc_2.reset()

class DECOLLE_LIFNet(nn.Module):
    def __init__(self,T:int,burnin:int,input_size:int,hidden_size:int,output_size:int,tau:float,surrogate_type:str='zo',
                 surrogate_param:float=0.5,criterion:Any=F.mse_loss) -> None:
        super(DECOLLE_LIFNet,self).__init__()
        self.T=T
        self.burnin=burnin
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=FixedFCBlock(input_size,hidden_size,output_size,tau,None,True,surrogate_type,surrogate_param)
        self.fc_2=FixedFCBlock(hidden_size,output_size,output_size,tau,None,True,surrogate_type,surrogate_param)
        self.criterion=criterion
    
    def step(self,input:torch.Tensor,labels:torch.Tensor,training:bool=False) -> tuple[torch.Tensor,torch.Tensor]:
        x=self.flatten(input)
        if training:
            loss_sum=0

            x,y_t=self.fc_1(x,True)
            loss_sum+=self.criterion(y_t.mean(1),labels)
            x=x.clone().detach()

            x,y_t=self.fc_2(self.flatten(x))
            loss_sum+=self.criterion(x.mean(1),labels)
            return x,loss_sum
        else:
            x,_=self.fc_1(x)
            x,_=self.fc_2(self.flatten(x))
            return x,self.criterion(x.mean(1),labels)
        
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        self.reset()
        if self.training:
            input=input.unsqueeze(1).repeat(1,+self.burnin+self.T,1,1,1)
            with torch.no_grad():
                self.step(input[:,:self.burnin,...],labels,True)
            x,loss_sum=self.step(input[:,self.burnin:,...],labels,True)
        else:
            input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
            x,loss_sum=self.step(input,labels,False)
        return x.mean(1),loss_sum
    
    def reset(self) -> None:
        self.fc_1.reset()
        self.fc_2.reset()

class WSDDTPC_LIFNet(nn.Module):
    def __init__(self,T:int,burn:int,input_size:int,hidden_size:int,output_size:int,tau:float,axon_tau:float,straight_through:bool=False,
                 surrogate_type:str='zo',surrogate_param:float=0.5,criterion:Any=F.mse_loss) -> None:
        super(WSDDTPC_LIFNet,self).__init__()
        self.T=T
        self.burn=burn
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=ConnectedFCBlock(input_size,hidden_size,output_size,tau,axon_tau,straight_through,surrogate_type,surrogate_param)
        self.fc_2=ConnectedFCBlock(hidden_size,output_size,output_size,tau,axon_tau,straight_through,surrogate_type,surrogate_param)
        self.criterion=criterion
    
    def step(self,input:torch.Tensor,labels:torch.Tensor,training:bool=False) -> tuple[torch.Tensor,torch.Tensor]:
        x=self.flatten(input)
        if training:
            labels=labels.unsqueeze(1).repeat(1,x.shape[1],1)
            loss_sum=0

            x,y_t=self.fc_1(x,labels)
            loss_sum+=self.criterion(x.mean(1),y_t.mean(1))
            x=x.clone().detach()

            x,_=self.fc_2(self.flatten(x),None)
            loss_sum+=self.criterion(x.mean(1),labels.mean(1))
            return x,loss_sum
        else:
            x,y_t=self.fc_1(x,None)
            x,y_t=self.fc_2(self.flatten(x),None)
            return x,self.criterion(x.mean(1),labels)
        
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        self.reset()
        if self.training:
            input=input.unsqueeze(1).repeat(1,+self.burn+self.T,1,1,1)
            with torch.no_grad():
                self.step(input[:,:self.burn,...],labels,True)
            x,loss_sum=self.step(input[:,self.burn:,...],labels,True)
        else:
            input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
            x,loss_sum=self.step(input,labels,False)
        return x.mean(1),loss_sum

    def reset(self) -> None:
        self.fc_1.reset()
        self.fc_2.reset()

# class WDRTP_LIFNet(nn.Module):
#     def __init__(self,T:int,input_size:int,hidden_size:int,output_size:int,tau:float,surrogate_type:str='zon',surrogate_param:float=0.5,
#                  criterion:Any=F.mse_loss) -> None:
#         super(WDRTP_LIFNet,self).__init__()
#         self.T=T
#         self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
#         self.fc_1=FCBlock(input_size,hidden_size,output_size,tau,surrogate_type,surrogate_param)
#         self.fc_2=FCBlock(hidden_size,output_size,output_size,tau,surrogate_type,surrogate_param)
#         self.criterion=criterion
        
#     def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
#         input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
#         x=self.flatten(input)
#         x=self.fc_1(x,labels)
#         x=self.fc_2(self.flatten(x),labels)
#         loss=self.criterion(x.mean(1),labels)
#         return x.mean(1),loss
#         # if self.training:
#         #     label=label.unsqueeze(1).repeat(1,self.T,1)
#         #     loss_sum=0
#         #     x,y_t=self.fc_1(x,label)
#         #     loss_sum+=self.criterion(x.mean(1),y_t.mean(1))
#         #     # print(y_t)
#         #     # self.fc_1.zero_grad()
#         #     x.require_grad=False
#         #     x,_=self.fc_2(self.flatten(x),None)
#         #     loss_sum+=self.criterion(x.mean(1),label.mean(1))
#         #     # self.fc_1.zero_grad()
#         #     x.require_grad=False
#         #     return x.mean(1),loss_sum
#         # else:
#         #     x,y_t=self.fc_1(x,None)
#         #     x,y_t=self.fc_2(self.flatten(x),None)
#         #     return x.mean(1),None