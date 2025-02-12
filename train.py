import torch
import neuron
import torch.nn.functional as F
from typing import Any
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import re
import logging

# train_data_loader=get_train_data_loader(batch_size=batch_size,shuffle=True)
# test_data_loader=get_test_data_loader(batch_size=batch_size,shuffle=True)

def train(model:torch.nn.Module,train_data_loader:DataLoader,test_data_loader:DataLoader,optimizer:Any,epochs:int,
          device:torch.device,surrogate_type:str,result_path:str) -> None:
    logging.info(f'Model structure:\n{model}')
    model.to(device)
    best_test_acc=0
    best_epoch=0
    train_acc_list=[]
    test_acc_list=[]
    train_loss_list=[]
    test_loss_list=[]
    for epoch in range(epochs):
        model.train()
        train_loss=0
        train_acc=0
        train_samples=0
        for img,label in tqdm(train_data_loader):
            # model.to(device)
            img=img.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            label_onehot=F.one_hot(label,10).float()

            output,loss=model(img,label_onehot)
            # output,loss_x,loss_y=model(img,label_onehot)
            loss.backward()
            optimizer.step()

            train_samples+=label.numel()
            train_loss+=loss.item()*label.numel()
            # train_loss+=loss_x.item()*label.numel()+loss_y.item()*label.numel()

            train_acc+=(output.argmax(1)==label).float().sum().item()
        test_loss,test_acc=evaluate(model,test_data_loader,device)
        if test_acc>best_test_acc:
            best_test_acc=test_acc
            best_epoch=epoch+1
            # torch.save(model.cpu().state_dict(),
            #            result_path+f'/result/{surrogate_type}_{epoch+1}_{test_loss}_{test_acc}.pth')
            first_str=[string for string in filter(lambda x:x!='',re.split(r'[.>\']+',str(type(model))))][-1]
            # for name in re.split(r'[.>\']',str(type(model))):
                # if name.find('SDDTP')!=-1:
                #     first_str=name
                #     break
            torch.save(model.cpu().state_dict(),
                       result_path+f'/result/{first_str}_{surrogate_type}_{epoch+1}_{test_loss}_{test_acc}.pth')
            model.to(device)
        train_loss/=train_samples
        train_acc/=train_samples
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        print('Epoch {:3d}: Train loss: {:4f} | Train acc: {:3f} | Test loss: {:4f} | Test acc: {:3f}'
              .format(epoch+1,train_loss,train_acc,test_loss,test_acc))
        logging.info('Epoch {:3d}: Train loss: {:4f} | Train acc: {:3f} | Test loss: {:4f} | Test acc: {:3f}'
              .format(epoch+1,train_loss,train_acc,test_loss,test_acc))
    print(f'Best test accuracy: {best_test_acc} at epoch {best_epoch}')
    logging.info(f'Best test accuracy: {best_test_acc} at epoch {best_epoch}')
    pd.DataFrame({"train_loss":train_loss_list,"test_loss":test_loss_list,"train_acc":train_acc_list,"test_acc":test_acc_list}).to_csv(
        result_path+'/result/curve.csv')

def evaluate(model:torch.nn.Module,test_data_loader:DataLoader,device:torch.device) -> tuple:
    model.eval()
    test_loss=0
    test_acc=0
    test_samples=0
    model.to(device)
    with torch.no_grad():
        for img,label in tqdm(test_data_loader):
            # model.to(device)
            img=img.to(device)
            label=label.to(device)
            label_onehot=F.one_hot(label,10).float()

            # if 'SDDTP' in str(type(model)):
            #     output,_=model(img,label_onehot)
            # else:
            #     output=model(img)
            output,loss=model(img,label_onehot)
            # if loss is None:
            #     loss=F.mse_loss(output,label_onehot)

            test_samples+=label.numel()
            test_loss+=loss.item()*label.numel()

            test_acc+=(output.argmax(1)==label).float().sum().item()
    
    return test_loss/test_samples,test_acc/test_samples