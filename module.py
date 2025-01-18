import torch
from neuron import LIFNode,IFNode
import torch.nn as nn
from torch.nn import functional as F

# class FCBlock(nn.Module):
#     def __init__(self,input_size:int,output_size:int,label_size,tau:float,surrogate_type:str='zon',surrogate_param:float=0.8) -> None:
#         super(FCBlock,self).__init__()
#         self.f_fc=nn.Linear(input_size,output_size,bias=False)
#         self.f_lif=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
#         self.t_fc=nn.Linear(label_size,output_size,bias=False)
#         self.t_lif=LIFNode(tau=tau,surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
#         # self.t_if=IFNode(surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
    
#     def forward(self,X:torch.Tensor,y_t:torch.Tensor) -> tuple[torch.Tensor]:
#         X=self.f_fc(X)
#         X=self.f_lif(X)
#         if y_t is not None:
#             y_t=self.t_fc(y_t)
#             y_t=self.t_lif(y_t)
#             # y_t=self.t_lif(y_t)
#             # y_t=F.relu(y_t)
#         return X,y_t

class FCBlock(nn.Module):
    def __init__(self,input_size:int,output_size:int,label_size:int,tau:float,surrogate_type:str='zon',surrogate_param:float=0.5) -> None:
        super(FCBlock,self).__init__()
        self.f_fc=nn.Linear(input_size,output_size,bias=False)
        self.f_lif=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        self.t_fc=nn.Parameter(torch.Tensor(size=(label_size,output_size)))
        torch.nn.init.kaiming_uniform_(self.t_fc)
        self.t_fc.requires_grad=False
        # self.t_lif=LIFNode(tau=tau,surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
    
    def forward(self,X:torch.Tensor,labels:torch.Tensor) -> tuple[torch.Tensor]:
        X=self.f_fc(X)
        X=self.f_lif(X)
        y_t=None
        if labels is not None:
            y_t=labels.matmul(self.t_fc).gt(0).float().detach()
        return X,y_t