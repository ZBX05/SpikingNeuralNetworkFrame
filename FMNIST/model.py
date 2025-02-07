import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron import LIFNode
from module import *
from typing import Any

        # self.conv_fc=nn.Sequential(
        #     # nn.Conv2d(1,channels,kernel_size=3,padding=1),
        #     # nn.BatchNorm2d(channels),
        #     # LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param),
        #     Conv2dBlock(1,channels,kernel_size=3,stride=1,padding=1,label_size=label_size,tau=tau,axon_tau=None,straight_through=False,
        #                 surrogate_type=surrogate_type,surrogate_param=surrogate_param),
        #     nn.MaxPool2d(2,2),  # 14 * 14

        #     # nn.Conv2d(channels,channels,kernel_size=3,padding=1),
        #     # nn.BatchNorm2d(channels),
        #     # LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param),
        #     Conv2dBlock(1,channels,kernel_size=3,stride=1,padding=1,label_size=label_size,tau=tau,axon_tau=None,straight_through=False,
        #                 surrogate_type=surrogate_type,surrogate_param=surrogate_param),
        #     nn.MaxPool2d(2,2),  # 7 * 7

        #     nn.Flatten(start_dim=2,end_dim=-1),
        #     # nn.Linear(channels*7*7,channels*4*4),
        #     # LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param),
        #     ConnectedFCBlock(channels*7*7,channels*4*4,label_size,tau,None,False,surrogate_type,surrogate_param),

        #     nn.Linear(channels*4*4,10),
        #     LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param),
        # )

class Base_CSNN(nn.Module):
    def __init__(self) -> None:
        super(Base_CSNN,self).__init__()
    
    def reset(self) -> None:
        if hasattr(self,'conv_1'):
            self.conv_1.reset()
        if hasattr(self,'conv_2'):
            self.conv_2.reset()
        if hasattr(self,'classifier'):
            self.classifier.reset()

class CSNN(Base_CSNN):
    def __init__(self,T:int,channels:int,label_size:int,tau:float,surrogate_type:str='zo',surrogate_param:float=0.5,
                 criterion:Any=F.mse_loss) -> None:
        super(CSNN,self).__init__()
        self.T=T
        self.criterion=criterion
        input_shape=(28,28)
        self.conv_1=Conv2dBlock(input_shape,1,channels,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,pool_stride=2,
                                pool_padding=0,label_size=label_size,tau=tau,axon_tau=None,straight_through=True,
                                surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        
        input_shape=(self.conv_1.output_height,self.conv_1.output_width)
        self.conv_2=Conv2dBlock(input_shape,channels,channels*2,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,
                                pool_stride=2,pool_padding=0,label_size=label_size,tau=tau,axon_tau=None,
                                straight_through=True,surrogate_type=surrogate_type,surrogate_param=surrogate_param)

        input_shape=(self.conv_2.output_height,self.conv_2.output_width)
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.classifier=FCBlock(channels*2*input_shape[0]*input_shape[1],10,None,tau,surrogate_type,surrogate_param)
    
    def step(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        x,_=self.conv_1(input)
        x,_=self.conv_2(x)
        x=self.flatten(x)
        # x,_=self.fc_1(x,None)
        x,_=self.classifier(x)
        return x.mean(1),self.criterion(x.mean(1),labels)
    
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> torch.Tensor:
        self.reset()
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        return self.step(input,labels)

class SDDTP_CSNN(Base_CSNN):
    def __init__(self,T:int,channels:int,label_size:int,tau:float,surrogate_type:str='zo',surrogate_param:float=0.5,
                 criterion:Any=F.mse_loss) -> None:
        super(SDDTP_CSNN,self).__init__()
        self.T=T
        self.criterion=criterion
        input_shape=(28,28)
        self.conv_1=Conv2dBlock(input_shape,1,channels,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,pool_stride=2,
                                pool_padding=0,label_size=label_size,tau=tau,axon_tau=None,straight_through=True,
                                surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        
        input_shape=(self.conv_1.output_height,self.conv_1.output_width)
        self.conv_2=Conv2dBlock(input_shape,channels,channels*2,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,
                                pool_stride=2,pool_padding=0,label_size=label_size,tau=tau,axon_tau=None,
                                straight_through=True,surrogate_type=surrogate_type,surrogate_param=surrogate_param)

        input_shape=(self.conv_2.output_height,self.conv_2.output_width)
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.classifier=FCBlock(channels*2*input_shape[0]*input_shape[1],10,None,tau,surrogate_type,surrogate_param)

    def step(self,input:torch.Tensor,labels:torch.Tensor,training:bool=True) -> torch.Tensor:
        x=input
        if training:
            labels=labels.unsqueeze(1).repeat(1,self.T,1)
            loss_sum=0

            x,y_t=self.conv_1(x,True)
            loss_sum+=self.criterion(y_t,labels)
            x=x.clone().detach()

            x,y_t=self.conv_2(x,True)
            loss_sum+=self.criterion(y_t,labels)
            x=x.clone().detach()

            x=self.flatten(x)
            # x,_=self.fc_1(x,None)
            x,_=self.classifier(x)
            loss_sum+=self.criterion(x.mean(1),labels.mean(1))
            return x.mean(1),loss_sum
        else:
            x,_=self.conv_1(x)
            x,_=self.conv_2(x)
            x=self.flatten(x)
            # x,_=self.fc_1(x,None)
            x,_=self.classifier(x)
            return x.mean(1),self.criterion(x.mean(1),labels)

    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> torch.Tensor:
        self.reset()
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        if self.training:
            return self.step(input,labels,True)
        else:
            return self.step(input,labels,False)

class AxonSDDTP_CSNN(Base_CSNN):
    def __init__(self,T:int,channels:int,label_size:int,tau:float,axon_tau:float,straight_through:bool=False,surrogate_type:str='zo',
                 surrogate_param:float=0.5,criterion:Any=F.mse_loss) -> None:
        super(AxonSDDTP_CSNN,self).__init__()
        self.T=T
        self.criterion=criterion
        input_shape=(28,28)
        self.conv_1=Conv2dBlock(input_shape,1,channels,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,pool_stride=2,
                                pool_padding=0,label_size=label_size,tau=tau,axon_tau=axon_tau,straight_through=straight_through,
                                surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        
        input_shape=(self.conv_1.output_height,self.conv_1.output_width)
        self.conv_2=Conv2dBlock(input_shape,channels,channels*2,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,
                                pool_stride=2,pool_padding=0,label_size=label_size,tau=tau,axon_tau=axon_tau,straight_through=straight_through,
                                surrogate_type=surrogate_type,surrogate_param=surrogate_param)

        input_shape=(self.conv_2.output_height,self.conv_2.output_width)
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.classifier=FCBlock(channels*2*input_shape[0]*input_shape[1],10,None,tau,surrogate_type,surrogate_param)

    def step(self,input:torch.Tensor,labels:torch.Tensor,training:bool=True) -> torch.Tensor:
        x=input
        if training:
            labels=labels.unsqueeze(1).repeat(1,self.T,1)
            loss_sum=0

            x,y_t=self.conv_1(x,True)
            loss_sum+=self.criterion(y_t,labels)
            x=x.clone().detach()

            x,y_t=self.conv_2(x,True)
            loss_sum+=self.criterion(y_t,labels)
            x=x.clone().detach()

            x=self.flatten(x)
            # x,_=self.fc_1(x,None)
            x,_=self.classifier(x)
            loss_sum+=self.criterion(x.mean(1),labels.mean(1))
            return x.mean(1),loss_sum
        else:
            x,_=self.conv_1(x)
            x,_=self.conv_2(x)
            x=self.flatten(x)
            # x,_=self.fc_1(x,None)
            x,_=self.classifier(x)
            return x.mean(1),self.criterion(x.mean(1),labels)

    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> torch.Tensor:
        self.reset()
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        if self.training:
            return self.step(input,labels,True)
        else:
            return self.step(input,labels,False)

class CASDDTP_CSNN(Base_CSNN):
    def __init__(self,T:int,channels:int,label_size:int,tau:float,axon_tau:float,straight_through:bool=False,surrogate_type:str='zo',
                 surrogate_param:float=0.5,criterion:Any=F.mse_loss) -> None:
        super(CASDDTP_CSNN,self).__init__()
        self.T=T
        self.criterion=criterion
        data_shape=(1,28,28)
        input_shape=(28,28)
        self.conv_1=ConnectedConv2dBlock(data_shape,input_shape,1,channels,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,
                                         pool_stride=2,pool_padding=0,label_size=label_size,tau=tau,axon_tau=axon_tau,
                                         straight_through=straight_through,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        
        input_shape=(self.conv_1.output_height,self.conv_1.output_width)
        self.conv_2=ConnectedConv2dBlock(data_shape,input_shape,channels,channels*2,conv_kernel_size=5,conv_stride=1,conv_padding=0,pool_kernel_size=2,
                                         pool_stride=2,pool_padding=0,label_size=label_size,tau=tau,axon_tau=axon_tau,
                                         straight_through=straight_through,surrogate_type=surrogate_type,surrogate_param=surrogate_param)

        input_shape=(self.conv_2.output_height,self.conv_2.output_width)
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.classifier=FCBlock(channels*2*input_shape[0]*input_shape[1],10,None,tau,surrogate_type,surrogate_param)

    def step(self,input:torch.Tensor,labels:torch.Tensor,training:bool=True) -> torch.Tensor:
        x=input
        if training:
            labels=labels.unsqueeze(1).repeat(1,self.T,1)
            loss_sum=0

            x,y_t=self.conv_1(x,input,True)
            loss_sum+=self.criterion(y_t,labels)
            x=x.clone().detach()

            x,y_t=self.conv_2(x,input,True)
            loss_sum+=self.criterion(y_t,labels)
            x=x.clone().detach()

            x=self.flatten(x)
            # x,_=self.fc_1(x,None)
            x,_=self.classifier(x)
            loss_sum+=self.criterion(x.mean(1),labels.mean(1))
            return x.mean(1),loss_sum
        else:
            x,_=self.conv_1(x)
            x,_=self.conv_2(x)
            x=self.flatten(x)
            # x,_=self.fc_1(x,None)
            x,_=self.classifier(x)
            return x.mean(1),self.criterion(x.mean(1),labels)

    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> torch.Tensor:
        self.reset()
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        if self.training:
            return self.step(input,labels,True)
        else:
            return self.step(input,labels,False)

class SDDTP_LIFNet(nn.Module):
    def __init__(self,T:int,input_size:int,hidden_size:int,output_size:int,tau:float,surrogate_type:str='zo',surrogate_param:float=0.5,
                 criterion:Any=F.mse_loss) -> None:
        super(SDDTP_LIFNet,self).__init__()
        self.T=T
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=FCBlock(input_size,hidden_size,output_size,tau,surrogate_type,surrogate_param)
        self.fc_2=FCBlock(hidden_size,output_size,output_size,tau,surrogate_type,surrogate_param)
        self.criterion=criterion
        
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        x=self.flatten(input)
        if self.training:
            labels=labels.unsqueeze(1).repeat(1,self.T,1)
            loss_sum=0

            x,y_t=self.fc_1(x,labels)
            loss_sum+=self.criterion(x.mean(1),y_t.mean(1))
            x.require_grad=False

            x,_=self.fc_2(self.flatten(x),None)
            loss_sum+=self.criterion(x.mean(1),labels.mean(1))
            x.require_grad=False
            return x.mean(1),loss_sum
        else:
            x,y_t=self.fc_1(x,None)
            x,y_t=self.fc_2(self.flatten(x),None)
            return x.mean(1),self.criterion(x.mean(1),labels)

class SDDTPC_LIFNet(nn.Module):
    def __init__(self,T:int,input_size:int,hidden_size:int,output_size:int,tau:float,surrogate_type:str='zo',surrogate_param:float=0.5,
                 criterion:Any=F.mse_loss) -> None:
        super(SDDTPC_LIFNet,self).__init__()
        self.T=T
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=ConnectedFCBlock(input_size,hidden_size,output_size,tau,2.0,False,surrogate_type,surrogate_param)
        self.fc_2=ConnectedFCBlock(hidden_size,output_size,output_size,tau,2.0,False,surrogate_type,surrogate_param)
        self.criterion=criterion
        
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        x=self.flatten(input)
        if self.training:
            labels=labels.unsqueeze(1).repeat(1,self.T,1)
            loss_sum=0

            x,y_t=self.fc_1(x,labels)
            loss_sum+=self.criterion(x.mean(1),y_t.mean(1))
            x.require_grad=False

            x,_=self.fc_2(self.flatten(x),None)
            loss_sum+=self.criterion(x.mean(1),labels.mean(1))
            x.require_grad=False
            return x.mean(1),loss_sum
        else:
            x,y_t=self.fc_1(x,None)
            x,y_t=self.fc_2(self.flatten(x),None)
            return x.mean(1),self.criterion(x.mean(1),labels)

class WSDDTPC_LIFNet(nn.Module):
    def __init__(self,T:int,input_size:int,hidden_size:int,output_size:int,tau:float,axon_tau:float,straight_through:bool=False,
                 surrogate_type:str='zo',surrogate_param:float=0.5,criterion:Any=F.mse_loss) -> None:
        super(WSDDTPC_LIFNet,self).__init__()
        self.T=T
        self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
        self.fc_1=ConnectedFCBlock(input_size,hidden_size,output_size,tau,axon_tau,straight_through,surrogate_type,surrogate_param)
        self.fc_2=ConnectedFCBlock(hidden_size,output_size,output_size,tau,axon_tau,straight_through,surrogate_type,surrogate_param)
        self.criterion=criterion
        
    def forward(self,input:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        input=input.unsqueeze(1).repeat(1,self.T,1,1,1)
        x=self.flatten(input)
        if self.training:
            labels=labels.unsqueeze(1).repeat(1,self.T,1)
            loss_sum=0

            x,y_t=self.fc_1(x,labels)
            loss_sum+=self.criterion(x.mean(1),y_t.mean(1))
            x.require_grad=False

            x,_=self.fc_2(self.flatten(x),None)
            loss_sum+=self.criterion(x.mean(1),labels.mean(1))
            x.require_grad=False
            return x.mean(1),loss_sum
        else:
            x,y_t=self.fc_1(x,None)
            x,y_t=self.fc_2(self.flatten(x),None)
            return x.mean(1),self.criterion(x.mean(1),labels)