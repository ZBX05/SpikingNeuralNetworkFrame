import torch
from torch import nn
from typing import Any

surrogate_type='sigmoid'
param=4.0

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input.gt(0).float() # spike=mem-self.v_threshold>0
    @staticmethod
    def backward(ctx,grad_output):
        input,=ctx.saved_tensors
        grad_input=grad_output.clone()
        if surrogate_type=='sigmoid':
            sgax=1/(1+torch.exp(-param*input))
            grad_surrogate=param*(1-sgax)*sgax
        elif surrogate_type=='arctan':
            grad_surrogate=param/(2*(1+torch.pow((torch.pi/2)*param*input,2)))
        return grad_surrogate.float()*grad_input

class LIFNode(nn.Module):
    # def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=0.5,activate_function:Any=ActFun.apply) -> None:
    def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=0.5,surrogate_type:str='sigmoid',surrogate_param:float=4.0) -> None:
        super(LIFNode, self).__init__()
        # self.act = F.sigmoid
        self.v_reset=v_reset
        self.v_threshold=v_threshold
        self.tau=tau
        self.activate_function=ActFun.apply
        # self.activate_function=self.ActFun.apply
        # self.surrogate_type=surrogate_type
        # self.surrogate_param=surrogate_param

    def forward(self,X:torch.Tensor):
        mem=0
        T=X.shape[1]
        spike_pot=[]
        for t in range(T):
            #***Charging function: H[t]=V[t-1]+1/tau*(X[t]-(V[t-1]-V_reset)) 0<tau<1***#
            if self.v_reset is not None:
                mem=mem+(X[:,t,...]+self.v_reset-mem)/self.tau
            else:
                mem=mem+(X[:,t,...]-mem)/self.tau
            # spike=self.activate_function(mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            spike=self.activate_function(mem-self.v_threshold)
            if self.v_reset is None:
                mem=mem-spike*self.v_threshold
            else:
                mem=(1-spike)*mem+self.v_reset*spike
            spike_pot.append(spike)
        return torch.stack(spike_pot,dim=1)
    
    class ActFun(torch.autograd.Function):
        @staticmethod
        def forward(ctx,input,surrogate_type,param):
            param=torch.tensor([param],device=input.device)
            if surrogate_type=='sigmoid':
                surrogate_type=torch.tensor([0])
            elif surrogate_type=='arctan':
                surrogate_type=torch.tensor([1])
            else:
                surrogate_type=torch.tensor([2])
            ctx.save_for_backward(input,param,surrogate_type)
            return input.gt(0).float() # spike=mem-self.v_threshold>0
        @staticmethod
        def backward(ctx,grad_output):
            input,param,surrogate_type=ctx.saved_tensors
            grad_input=grad_output.clone()
            surrogate_type=surrogate_type.item()
            # sigmoid
            if surrogate_type==0:
                sgax=1/(1+torch.exp(-param*input))
                grad_surrogate=param*(1-sgax)*sgax
            # arctan
            elif surrogate_type==1:
                grad_surrogate=param/(2*(1+torch.pow((torch.pi/2)*param*input,2)))
            return grad_surrogate.float()*grad_input,None,None
