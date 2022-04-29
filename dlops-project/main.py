import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import config
from Customdata import get_dataloader
from model import return_models, train, return_optimizer_loss

# def test_model(net_g,val_data):
#     net_g.eval()
#     x,y = next(iter(val_data))
#     x = x[:10]
#     y = y[:10]
#     x = x.to(config.device)
#     y = y.to(config.device)
#     y_gene = net_g(x)
#     image_grid = torch.cat([x,y,y_gene],dim=3)
#     image_grid = image_grid.permute(0,2,3,1).flatten(start_dim=0,end_dim=1).detach().cpu().numpy()
#     plt.figure(figsize=(5,4))
#     plt.imshow(image_grid)
#     plt.savefig(config.save_test_imgs,format='svg')
#     plt.show()
#     # print(image_grid.shape)
    
def test_model(net_g,val_data):
    net_g.eval()
    x,y = next(iter(val_data))
    x = x.to(config.device)
    y = y.to(config.device)
    for i in range(15):
        x_i = x[i].unsqueeze(0)
        y_i = y[i].unsqueeze(0)
        y_gen = net_g(x_i)
        image_grid = torch.cat([x_i,y_i,y_gen],dim=3)
        image_grid = image_grid.permute(0,2,3,1).flatten(start_dim=0,end_dim=1).detach().cpu().numpy()
        plt.figure(figsize=(10,8))
        plt.imshow(image_grid)
        plt.savefig(f'{i+1}_example.svg',format='svg')
        plt.show()


def plotloss(history):
    len_values = len(history['genloss'])
    plt.figure(figsize=(5,4))
    plt.plot(list(range(1,len_values + 1)),history['genloss'], label = 'genloss')
    plt.plot(list(range(1,len_values + 1)),history['discloss'], label = 'discloss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(config.save_loss_path,format='svg')
    plt.show()

def pipeline():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    # os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id
    
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(config.device)

    train_data, val_data = get_dataloader()
    print(len(train_data))
    print(len(val_data))

    gene,disc = return_models()
    ogen, odis, bce,l1_loss,g_scalar,d_scalar = return_optimizer_loss(gene,disc)

    history = train(gene,disc,train_data,bce,l1_loss,ogen,odis,g_scalar,d_scalar,n_epochs=config.epochs)
    plotloss(history)
    test_model(gene,val_data)
    
    # torch.save(gene.state_dict(),config.model_path)


if __name__ == '__main__':
    pipeline()