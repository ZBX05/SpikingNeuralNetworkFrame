import torch
from torch import nn
from function import ActFun,HookFunction

class LIFNode(nn.Module):
    # def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=0.5,activate_function:Any=ActFun.apply) -> None:
    def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=0.5,surrogate_type:str='zon',surrogate_param:float=0.5) -> None:
        super(LIFNode, self).__init__()
        # self.act = F.sigmoid
        self.v_reset=v_reset
        self.v_threshold=v_threshold
        self.tau=tau
        self.activate_function=ActFun.apply
        self.surrogate_type=surrogate_type
        self.surrogate_param=surrogate_param
        # self.activate_function=self.ActFun.apply
        # self.surrogate_type=surrogate_type
        # self.surrogate_param=surrogate_param

    def forward(self,X:torch.Tensor) -> torch.Tensor:
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
            spike=self.activate_function(mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            if self.v_reset is None:
                mem=mem-spike*self.v_threshold
            else:
                mem=(1-spike)*mem+self.v_reset*spike
            spike_pot.append(spike)
        return torch.stack(spike_pot,dim=1)

class IFNode(nn.Module):
    # def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=0.5,activate_function:Any=ActFun.apply) -> None:
    def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,surrogate_type:str='zon',surrogate_param:float=0.5) -> None:
        super(IFNode, self).__init__()
        # self.act = F.sigmoid
        self.v_reset=v_reset
        self.v_threshold=v_threshold
        self.activate_function=ActFun.apply
        self.surrogate_type=surrogate_type
        self.surrogate_param=surrogate_param
        # self.activate_function=self.ActFun.apply
        # self.surrogate_type=surrogate_type
        # self.surrogate_param=surrogate_param

    def forward(self,X:torch.Tensor) -> torch.Tensor:
        mem=0
        T=X.shape[1]
        spike_pot=[]
        for t in range(T):
            #***Charging function: H[t]=V[t-1]+1/tau*(X[t]-(V[t-1]-V_reset)) 0<tau<1***#
            mem=mem+X[:,t,...]
            # if self.v_reset is not None:
            #     mem=mem+(X[:,t,...]+self.v_reset-mem)/self.tau
            # else:
            #     mem=mem+(X[:,t,...]-mem)/self.tau
            # spike=self.activate_function(mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            spike=self.activate_function(mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            if self.v_reset is None:
                mem=mem-spike*self.v_threshold
            else:
                mem=(1-spike)*mem+self.v_reset*spike
            spike_pot.append(spike)
        return torch.stack(spike_pot,dim=1)
