from torch import nn

def get_conv2d_output_dim(conv2d:nn.Module,input_size:int) -> int:
    padding=conv2d.conv_padding
    kernel_size=conv2d.conv_kernel_size
    stride=conv2d.conv_stride
    output_dim=(input_size+2*padding-kernel_size)//stride+1
    return output_dim

def get_maxpool2d_output_dim(maxpool2d:nn.Module,input_size:int) -> int:
    kernel_size=maxpool2d.pool_kernel_size
    padding=maxpool2d.pool_padding
    stride=maxpool2d.pool_stride
    output_dim=(input_size+2*padding-kernel_size)//stride+1
    return output_dim

def get_conv2d_block_output_dim(conv2d:nn.Module,input_size:int) -> int:
    return get_maxpool2d_output_dim(conv2d,get_conv2d_output_dim(conv2d,input_size))