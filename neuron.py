import torch
from torch import nn
from function import ActFun
import math

class BaseNode(nn.Module):
    def __init__(self) -> None:
        super(BaseNode,self).__init__()

    def reset(self) -> None:
        self.mem=0
        self.spike_pot=[]
        if hasattr(self,'out_i'):
            self.out_i=0
        if hasattr(self,'i_pot'):
            self.i_pot=[]

class LIFNode(BaseNode):
    # def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=0.5,activate_function:Any=ActFun.apply) -> None:
    def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=2,surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(LIFNode,self).__init__()
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
        self.mem=0
        self.spike_pot=[]

    def forward(self,X:torch.Tensor) -> torch.Tensor:
        T=X.shape[1]
        for t in range(T):
            #***Charging function: H[t]=V[t-1]+1/tau*(X[t]-(V[t-1]-V_reset)) 0<tau<1***#
            if self.v_reset is not None:
                self.mem=self.mem+(X[:,t,...]+self.v_reset-self.mem)/self.tau
            else:
                self.mem=self.mem+(X[:,t,...]-self.mem)/self.tau
            # spike=self.activate_function(mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            spike=self.activate_function(self.mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            if self.v_reset is None:
                self.mem=self.mem-spike*self.v_threshold
            else:
                self.mem=(1-spike)*self.mem+self.v_reset*spike
            self.spike_pot.append(spike)
        return torch.stack(self.spike_pot,dim=1)

class AxonLIFNode(BaseNode):
    def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=2,axon_tau:float=2,straight_through:bool=False,surrogate_type:str='zo',
                 surrogate_param:float=0.5) -> None:
        super(AxonLIFNode,self).__init__()
        # self.act = F.sigmoid
        self.v_reset=v_reset
        self.v_threshold=v_threshold
        self.tau=tau
        self.straight_through=straight_through
        self.activate_function=ActFun.apply
        self.surrogate_type=surrogate_type
        self.surrogate_param=surrogate_param
        init_w=-math.log(axon_tau-1)
        self.w=nn.Parameter(torch.as_tensor(init_w),requires_grad=True)
        # self.activate_function=self.ActFun.apply
        # self.surrogate_type=surrogate_type
        # self.surrogate_param=surrogate_param
        self.mem=0
        self.spike_pot=[]
        self.out_i=0
        self.i_pot=[]
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        T=X.shape[1]
        for t in range(T):
            #***Charging function: H[t]=V[t-1]+1/tau*(X[t]-(V[t-1]-V_reset)) 0<tau<1***#
            if self.v_reset is not None:
                self.mem=self.mem+(X[:,t,...]+self.v_reset-self.mem)/self.tau
            else:
                self.mem=self.mem+(X[:,t,...]-self.mem)/self.tau
            # spike=self.activate_function(mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            spike=self.activate_function(self.mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            if self.v_reset is None:
                self.mem=self.mem-spike*self.v_threshold
            else:
                self.mem=(1-spike)*self.mem+self.v_reset*spike
            self.spike_pot.append(spike)
            if not self.straight_through:
                inv_tau=self.w.sigmoid()
                # out_i=(out_i-(1-X[:,t,...])*out_i)*inv_tau+X[:,t,...]
                self.out_i=self.out_i*inv_tau+spike
                self.i_pot.append(self.out_i)
            else:
                self.i_pot=self.spike_pot
        return torch.stack(self.spike_pot,dim=1),torch.stack(self.i_pot,dim=1)

class IFNode(BaseNode):
    # def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,tau:float=0.5,activate_function:Any=ActFun.apply) -> None:
    def __init__(self,v_threshold:float=1.0,v_reset:float=0.0,surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(IFNode,self).__init__()
        # self.act = F.sigmoid
        self.v_reset=v_reset
        self.v_threshold=v_threshold
        self.activate_function=ActFun.apply
        self.surrogate_type=surrogate_type
        self.surrogate_param=surrogate_param
        # self.activate_function=self.ActFun.apply
        # self.surrogate_type=surrogate_type
        # self.surrogate_param=surrogate_param
        self.mem=0
        self.spike_pot=[]

    def forward(self,X:torch.Tensor) -> torch.Tensor:
        T=X.shape[1]
        for t in range(T):
            #***Charging function: H[t]=V[t-1]+X[t]***#
            self.mem=self.mem+X[:,t,...]
            # if self.v_reset is not None:
            #     mem=mem+(X[:,t,...]+self.v_reset-mem)/self.tau
            # else:
            #     mem=mem+(X[:,t,...]-mem)/self.tau
            # spike=self.activate_function(mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            spike=self.activate_function(self.mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            if self.v_reset is None:
                self.mem=self.mem-spike*self.v_threshold
            else:
                self.mem=(1-spike)*self.mem+self.v_reset*spike
            self.spike_pot.append(spike)
        return torch.stack(self.spike_pot,dim=1)
class AxonUnit(nn.Module):
    def __init__(self,tau:float=2,straight_through:bool=False) -> None:
        super(AxonUnit,self).__init__()
        self.straight_through=straight_through
        self.tau=tau
        init_w=-math.log(tau-1)
        self.w=nn.Parameter(torch.as_tensor(init_w),requires_grad=True)
        
        self.i_pot=[]
        self.out_i=0
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        if self.straight_through:
            return X
        T=X.shape[1]
        i_pot=[]
        out_i=0
        for t in range(T):
            inv_tau=self.w.sigmoid()
            # out_i=(out_i-(1-X[:,t,...])*out_i)*inv_tau+X[:,t,...]
            out_i=out_i*inv_tau+X[:,t,...]
            i_pot.append(out_i)
        return torch.stack(i_pot,dim=1)
    
    def reset(self) -> None:
        pass