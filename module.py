import torch
from neuron import *
import torch.nn as nn
from torch.nn import functional as F
from function import ModifyFunction
import numpy as np

class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    # This container enables seq input to layers contained in it(e.g. [B,T,C,H,W] -> nn.Conv2d).
    def __init__(self,*args:nn.Module) -> None:
        super(SeqToANNContainer,self).__init__()
        if len(args)==1:
            self.module=args[0]
        else:
            self.module=nn.Sequential(*args)

    def forward(self,x_seq:torch.Tensor) -> torch.Tensor:
        y_shape=[x_seq.shape[0],x_seq.shape[1]]
        y_seq=self.module(x_seq.flatten(0,1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class BaseBlock(nn.Module):
    def __init__(self) -> None:
        super(BaseBlock,self).__init__()
    
    def reset(self) -> None:
        if hasattr(self,'f_lif'):
            self.f_lif.reset()
        if hasattr(self,'t_lif'):
            self.t_lif.reset()
        if hasattr(self,'axon'):
            self.axon.reset()

    def set_t_layer_parameters(self,layer:nn.Module,lc_ampl:float) -> None:
        stdv=lc_ampl/np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv,stdv)
        self.set_t_layer_bias_parameters(layer,lc_ampl)

    def set_t_layer_bias_parameters(self,layer:nn.Module,lc_ampl:float) -> None:
        stdv=lc_ampl/np.sqrt(layer.weight.size(1))
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv,stdv)


class FCBlock(BaseBlock):
    def __init__(self,input_size:int,output_size:int,label_size:int,tau:float,surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(FCBlock,self).__init__()
        self.f_fc=nn.Linear(input_size,output_size,bias=True)
        self.f_lif=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        if label_size is not None:
            self.t_fc=nn.Linear(output_size,label_size,bias=True)
            self.set_t_layer_bias_parameters(self.t_fc,0.5)
            self.t_lif=LIFNode(tau=tau,surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
    
    def forward(self,X:torch.Tensor,target:bool=False) -> tuple[torch.Tensor,torch.Tensor]:
        X=self.f_fc(X)
        X=self.f_lif(X)
        y_t=None
        if target:
            y_t=self.t_fc(X)
            y_t=self.t_lif(y_t)
        return X,y_t

class ConnectedFCBlock(BaseBlock):
    def __init__(self,input_size:int,output_size:int,label_size:int,tau:float,axon_tau:float=None,straight_through:bool=False,
                 surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(ConnectedFCBlock,self).__init__()
        self.f_fc=nn.Linear(input_size,output_size,bias=True)
        self.f_lif=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        if label_size is not None:
            self.t_fc=nn.Linear(output_size,label_size,bias=True)
            self.t_lif=LIFNode(tau=tau,surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
            if axon_tau is not None:
                self.f_lif=AxonLIFNode(tau=tau,axon_tau=axon_tau,straight_through=straight_through,
                                       surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
                # self.c_yt=nn.Parameter(torch.Tensor(size=(1,output_size)),requires_grad=True)
                # self.c_ax=nn.Parameter(torch.Tensor(size=(1,output_size)),requires_grad=True)
                # torch.nn.init.xavier_uniform_(self.c_yt)
                # torch.nn.init.xavier_uniform_(self.c_ax)
                self.c_yt=nn.Parameter(torch.Tensor(torch.ones(size=(1,output_size))),requires_grad=True)
                self.c_ax=nn.Parameter(torch.Tensor(torch.ones(size=(1,output_size))),requires_grad=True)

                # self.x_random_matrix=nn.Parameter(torch.rand(size=(1,output_size)),requires_grad=False)
            
    
    def forward(self,X:torch.Tensor,target:bool=False) -> tuple[torch.Tensor,torch.Tensor]:
        X=self.f_fc(X)
        X=self.f_lif(X)
        y_t=None
        if target:
            y_t=self.t_fc(X)
        return X,y_t

class FixedFCBlock(BaseBlock):
    def __init__(self,input_size:int,output_size:int,label_size:int,tau:float,axon_tau:float=None,straight_through:bool=False,
                 surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(FixedFCBlock,self).__init__()
        self.f_fc=nn.Linear(input_size,output_size,bias=True)
        self.f_lif=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        self.t_fc=nn.Linear(output_size,label_size,bias=True)
        for param in self.t_fc.parameters():
            param.requires_grad=False
        self.set_t_layer_parameters(self.t_fc,0.5)
    
    def forward(self,X:torch.Tensor,target:bool=False) -> tuple[torch.Tensor,torch.Tensor]:
        X=self.f_fc(X)
        X=self.f_lif(X)
        y_t=None
        if target:
            y_t=self.t_fc(X)
        return X,y_t

class ConvBaseBlock(BaseBlock):
    def __init__(self,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,conv_padding:int,
                 pool_kernel_size:int,pool_stride:int,pool_padding:int,label_size:int,tau:float,axon_tau:float=None,
                 straight_through:bool=False,surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(ConvBaseBlock,self).__init__()
        self.conv_kernel_size=conv_kernel_size
        self.conv_stride=conv_stride
        self.conv_padding=conv_padding
        self.pool_kernel_size=pool_kernel_size
        self.pool_stride=pool_stride
        self.pool_padding=pool_padding

        self.conv2d_layer=SeqToANNContainer(
            nn.Conv2d(in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding),
            nn.BatchNorm2d(out_channels),
        )
        self.f_lif=LIFNode(tau=tau,surrogate_type=surrogate_type,surrogate_param=surrogate_param)
        self.maxpool2d_layer=SeqToANNContainer(
            nn.MaxPool2d(pool_kernel_size,pool_stride,pool_padding)
        )

        input_height=input_shape[0]
        input_width=input_shape[1]
        self.conv2d_output_height=self.get_conv2d_output_dim(input_height)
        self.conv2d_output_width=self.get_conv2d_output_dim(input_width)
        self.output_height,self.output_width=self.get_conv2d_block_output_dim(input_height,input_width)
        if label_size is not None:
            self.flatten=nn.Flatten(start_dim=2,end_dim=-1)
            self.t_fc=nn.Linear(out_channels*self.output_height*self.output_width,label_size,bias=True)
            self.set_t_layer_parameters(self.t_fc,0.5)
            self.t_lif=LIFNode(tau=tau,surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
            if axon_tau is not None:
                self.f_lif=AxonLIFNode(tau=tau,axon_tau=axon_tau,straight_through=straight_through,
                                       surrogate_type='zo' if surrogate_type=='zon' else surrogate_type,surrogate_param=surrogate_param)
                # self.axon=AxonUnit(tau=axon_tau,straight_through=straight_through)
                # self.c_yt=nn.Parameter(torch.Tensor(torch.ones(size=(1,out_channels*self.conv2d_output_height*self.conv2d_output_height))),
                #                        requires_grad=True)
                # self.c_ax=nn.Parameter(torch.Tensor(torch.ones(size=(1,out_channels*self.conv2d_output_height*self.conv2d_output_height))),
                #                        requires_grad=True)
        
    def get_conv2d_output_dim(self,dim_size:int) -> int:
        padding=self.conv_padding
        kernel_size=self.conv_kernel_size
        stride=self.conv_stride
        output_dim=(dim_size+2*padding-kernel_size)//stride+1
        return output_dim

    def get_maxpool2d_output_dim(self,dim_size:int) -> int:
        kernel_size=self.pool_kernel_size
        padding=self.pool_padding
        stride=self.pool_stride
        output_dim=(dim_size+2*padding-kernel_size)//stride+1
        return output_dim

    def get_conv2d_block_output_dim(self,input_height:int,input_width:int) -> tuple[int,int]:
        output_height=self.get_maxpool2d_output_dim(self.get_conv2d_output_dim(input_height))
        output_width=self.get_maxpool2d_output_dim(self.get_conv2d_output_dim(input_width))
        return output_height,output_width

class FixedConv2dBlock(ConvBaseBlock):
    def __init__(self,input_shape:int,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,conv_padding:int,pool_kernel_size:int,
                 pool_stride:int,pool_padding:int,label_size:int,tau:float,axon_tau:float=None,straight_through:bool=False,
                 surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(FixedConv2dBlock,self).__init__(input_shape,in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,pool_kernel_size,
                                              pool_stride,pool_padding,label_size,tau,None,True,surrogate_type,surrogate_param)
        for param in self.t_fc.parameters():
            param.requires_grad=False
    
    def forward(self,X:torch.Tensor,target:bool=False) -> tuple[torch.Tensor]:
        X=self.conv2d_layer(X)
        X=self.f_lif(X)
        X=self.maxpool2d_layer(X)
        y_t=None
        if target:
            y_t=self.t_fc(self.flatten(X))
        return X,y_t

class Conv2dBlock(ConvBaseBlock):
    def __init__(self,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,conv_padding:int,
                 pool_kernel_size:int,pool_stride:int,pool_padding:int,label_size:int,tau:float,axon_tau:float=None,
                 straight_through:bool=False,surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(Conv2dBlock,self).__init__(input_shape,in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,pool_kernel_size,
                                         pool_stride,pool_padding,label_size,tau,axon_tau,straight_through,surrogate_type,surrogate_param)
    
    def forward(self,X:torch.Tensor,target:bool=False) -> tuple[torch.Tensor]:
        self.reset()
        X=self.conv2d_layer(X)
        if isinstance(self.f_lif,AxonLIFNode):
            X,c_X=self.f_lif(X)
            X=self.maxpool2d_layer(X)
            c_X=self.maxpool2d_layer(c_X)
        else:
            X=self.f_lif(X)
            X=self.maxpool2d_layer(X)
        y_t=None
        if target and isinstance(self.f_lif,AxonLIFNode):
            y_t=self.t_fc(self.flatten(c_X))
            y_t=self.t_lif(y_t)
            # y_t=y_t.unsqueeze(-1).unsqueeze(-1)
            # # c_X=self.flatten(X.clone().detach())
            # c_X=X.clone().detach()
            # c_X=self.c_ax.mul(self.axon(c_X))
            # y_t=self.t_conv2d_layer(y_t)
            # # y_t=self.t_fc(y_t)
            # y_t=self.c_yt.mul(y_t)
            # y_t=self.t_lif(y_t+c_X)
            # # y_t=y_t.view_as(X)
            # # y_t=self.t_lif(y_t+X)
            # # y_t=F.relu(y_t)
        elif target and not isinstance(self.f_lif,AxonLIFNode):
            y_t=self.t_fc(self.flatten(X))
            y_t=self.t_lif(y_t)
        return X,y_t

class ConnectedConv2dBlock(ConvBaseBlock):
    def __init__(self,data_shape:tuple,input_shape:tuple,in_channels:int,out_channels:int,conv_kernel_size:int,conv_stride:int,
                 conv_padding:int,pool_kernel_size:int,pool_stride:int,pool_padding:int,label_size:int,tau:float,axon_tau:float=None,
                 straight_through:bool=False,surrogate_type:str='zo',surrogate_param:float=0.5) -> None:
        super(ConnectedConv2dBlock,self).__init__(input_shape,in_channels,out_channels,conv_kernel_size,conv_stride,conv_padding,
                                                  pool_kernel_size,pool_stride,pool_padding,label_size,tau,axon_tau,straight_through,
                                                  surrogate_type,surrogate_param)

        self.jump_connect=SeqToANNContainer(
            nn.Conv2d(data_shape[0],out_channels,kernel_size=self.get_conv2d_kernel_size(data_shape,self.conv2d_output_height,
                                                                                         self.conv2d_output_width),stride=1,padding=0)
        )
        # self.jump_connect=nn.Linear(data_dim,out_channels*self.output_height*self.output_width)
        # for param in self.jump_connect.parameters():
        #     param.requires_grad=False
        # self.set_t_layer_parameters(self.jump_connect,0.5)
    
    def get_conv2d_kernel_size(self,data_shape:tuple,output_height:int,output_width:int) -> tuple[int,int]:
        input_height,input_width=data_shape[1],data_shape[2]
        padding=0
        stride=1
        kernel_height=input_height-(output_height-1)*stride-2*padding
        kernel_width=input_width-(output_width-1)*stride-2*padding
        return kernel_height,kernel_width
    
    def forward(self,X:torch.Tensor,input:torch.Tensor=None,target:bool=False) -> tuple[torch.Tensor,torch.Tensor]:
        self.reset()
        X=self.conv2d_layer(X)
        jump_X=self.jump_connect(input)
        X=X+jump_X
        if isinstance(self.f_lif,AxonLIFNode):
            X,c_X=self.f_lif(X)
            X=self.maxpool2d_layer(X)
            c_X=self.maxpool2d_layer(c_X)
        else:
            X=self.f_lif(X)
            X=self.maxpool2d_layer(X)
        y_t=None
        if target and isinstance(self.f_lif,AxonLIFNode):
            # jump_X=self.jump_connect(input.reshape(input.shape[0],input.shape[1],-1))
            y_t=self.t_fc(self.flatten(c_X))
            y_t=self.t_lif(y_t)
            # y_t=y_t.unsqueeze(-1).unsqueeze(-1)
            # # c_X=self.flatten(X.clone().detach())
            # c_X=X.clone().detach()
            # c_X=self.c_ax.mul(self.axon(c_X))
            # y_t=self.t_conv2d_layer(y_t)
            # # y_t=self.t_fc(y_t)
            # y_t=self.c_yt.mul(y_t)
            # y_t=self.t_lif(y_t+c_X)
            # # y_t=y_t.view_as(X)
            # # y_t=self.t_lif(y_t+X)
            # # y_t=F.relu(y_t)
        elif target and not isinstance(self.f_lif,AxonLIFNode):
            # jump_X=self.jump_connect(input.reshape(input.shape[0],input.shape[1],-1))
            y_t=self.t_fc(self.flatten(X))
            y_t=self.t_lif(y_t)
        return X,y_t
    
class MaxPool2dBlock(nn.Module):
    def __init__(self,kernel_size:int,stride:int,padding:int=0) -> None:
        super(MaxPool2dBlock,self).__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.maxpool2d_layer=SeqToANNContainer(nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding))
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        return self.maxpool2d_layer(X)