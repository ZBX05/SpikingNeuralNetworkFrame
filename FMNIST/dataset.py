from torchvision import datasets,transforms
import os,sys
from torch.utils.data import DataLoader

root_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

train_data=datasets.FashionMNIST(root=root_path+"/data",train=True,transform=transforms.ToTensor(),download=True)
test_data=datasets.FashionMNIST(root=root_path+"/data",train=False,transform=transforms.ToTensor(),download=True)

def get_train_data_loader(batch_size:int,shuffle:bool) -> DataLoader:
    train_data=datasets.FashionMNIST(root=root_path+'/data',train=True,transform=transforms.ToTensor(),download=False)
    return DataLoader(dataset=train_data,batch_size=batch_size,shuffle=shuffle)

def get_test_data_loader(batch_size:int,shuffle:bool) -> DataLoader:
    test_data=datasets.FashionMNIST(root=root_path+'/data',train=False,transform=transforms.ToTensor(),download=False)
    return DataLoader(dataset=test_data,batch_size=batch_size,shuffle=shuffle)