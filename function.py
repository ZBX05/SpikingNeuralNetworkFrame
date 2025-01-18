import torch
from numpy import prod

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,surrogate_type,param):
        param=param=torch.tensor([param],device=input.device)
        ctx.save_for_backward(input,param)
        ctx.in_1=surrogate_type
        return input.gt(0).float() # spike=mem-self.v_threshold>0

    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        input,param=ctx.saved_tensors
        surrogate_type=ctx.in_1
        param=param.item()
        if surrogate_type=='sigmoid':
            sgax=1/(1+torch.exp(-param*input))
            grad_surrogate=param*(1-sgax)*sgax
        elif surrogate_type=='arctan':
            grad_surrogate=param/(2*(1+torch.pow((torch.pi/2)*param*input,2)))
        elif surrogate_type=='zo' or surrogate_type =='zon': # or surrogate_type=='zos':
            sample_size=5
            abs_z=torch.abs(torch.randn((sample_size,)+input.size(),device=input.device,dtype=torch.float))
            if surrogate_type=='zo':
                t=torch.abs(input[None,:,:])<abs_z*param
                grad_surrogate=torch.mean(t*abs_z,dim=0)/(2*param)
            elif surrogate_type=='zon':
                c=torch.normal(0,torch.sqrt(input[None,:,:].std(dim=1)))
                t=torch.abs(input[None,:,:])<(abs_z*param+c)
                t=t & ((-abs_z*param+c)<torch.abs(input[None,:,:]))
                grad_surrogate=torch.mean(t*abs_z,dim=0)/(2*param)
            # elif surrogate_type=='zos':
            #     # t=-abs_z*param<torch.abs(input[None,:,:])
            #     # t=t & (torch.abs(input[None,:,:])<0)
            #     # t=torch.abs(input[None,:,:])<(abs_z*param+torch.normal(0,torch.sqrt(input[None,:,:].std(dim=1))))
            #     # t=torch.abs(input[None,:,:])<(abs_z*param+torch.randn_like(abs_z,device=input.device))
            #     t=torch.abs(input[None,:,:])<abs_z*param
            #     grad_surrogate=torch.mean(t*abs_z,dim=0)/(1*param)
        else:
            raise NameError('Error: surrogate type '+str(surrogate_type)+' is not supported')
        return grad_surrogate.float()*grad_input,None,None

class HookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,labels,fb_weights,train_mode):
        if train_mode in ['DRTP','WDRTP']:
            ctx.save_for_backward(input,labels,fb_weights)
        ctx.in_1=train_mode
        return input
    
    @staticmethod
    def backward(ctx,grad_output):
        train_mode=ctx.in_1
        if train_mode == 'BP' or train_mode=='SDDTP':
            return grad_output,None,None,None
        grad_input=grad_output.clone()
        input,labels,fb_weights=ctx.saved_tensors
        if train_mode=='DRTP' or train_mode=='WDRTP':
            grad_input=labels.mm(fb_weights.view(-1,prod(fb_weights.shape[1:]))).view(grad_input.shape)
            # print(labels,grad_input,grad_input.shape)
            # exit()
        # grad_input=labels.mm(fb_weights.view(-1,prod(fb_weights.shape[1:]))).view(input.shape)
        return grad_input,None,None,None