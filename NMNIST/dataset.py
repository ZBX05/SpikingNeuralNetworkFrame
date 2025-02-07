from torchvision import datasets,transforms
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader
import os
import sys

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

train_data=NMNIST(root=root_path+"\\data",train=True,data_type='frame',frames_number=20,split_by='number',transform=transforms.ToTensor())
test_data=NMNIST(root=root_path+"\\data",train=False,data_type='frame',frames_number=20,split_by='number',transform=transforms.ToTensor())

def get_train_data_loader(batch_size:int,shuffle:bool) -> DataLoader:
    train_data=NMNIST(root=root_path+'\\data',train=True,transform=transforms.ToTensor(),download=False)
    return DataLoader(dataset=train_data,batch_size=batch_size,shuffle=shuffle)

def get_test_data_loader(batch_size:int,shuffle:bool) -> DataLoader:
    test_data=NMNIST(root=root_path+'\\data',train=False,transform=transforms.ToTensor(),download=False)
    return DataLoader(dataset=test_data,batch_size=batch_size,shuffle=shuffle)