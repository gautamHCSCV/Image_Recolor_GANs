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
# from model import return_models

class block(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,act="relu",use_dropout=False):
        super(block,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,2,1,bias=False,padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2)
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class generator(nn.Module):
    def __init__(self,in_channels=3,features=64):
        super(generator,self).__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels,features,4,2,1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.down1 = block(features,features*2,down=True,act="Leaky")
        self.down2 = block(features*2,features*4,down=True,act="Leaky")
        self.down3 = block(features*4,features*8,down=True,act="Leaky")
        self.down4 = block(features*8,features*8,down=True,act="Leaky")
        self.down5 = block(features*8,features*8,down=True,act="Leaky")
        self.down6 = block(features*8,features*8,down=True,act="Leaky")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1,padding_mode="reflect"),nn.ReLU()
        )

        self.up1 = block(features*8,features*8,down=False,act='relu',use_dropout=True)
        self.up2 = block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
        self.up3 = block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
        self.up4 = block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
        self.up5 = block(features*8*2,features*4,down=False,act='relu',use_dropout=True)
        self.up6 = block(features*4*2,features*2,down=False,act='relu',use_dropout=True)
        self.up7 = block(features*2*2,features,down=False,act='relu',use_dropout=True)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2,in_channels,4,2,1),
            nn.Tanh()
        )

    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d7],dim=1))
        up3 = self.up3(torch.cat([up2,d6],dim=1))
        up4 = self.up4(torch.cat([up3,d5],dim=1))
        up5 = self.up5(torch.cat([up4,d4],dim=1))
        up6 = self.up6(torch.cat([up5,d3],dim=1))
        up7 = self.up7(torch.cat([up6,d2],dim=1))

        return self.final_up(torch.cat([up7,d1],dim=1))

def return_models():
    gene = generator()
    gene = gene.to(config.device)
    return gene

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# api_key = 'ae12d9032b94bfedc39f2e1beacfbf9909359ffc'
# os.environ['WANDB_API_KEY'] = api_key 
# os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id

torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(config.device)

train_data, val_data = get_dataloader()
print(len(train_data))
print(len(val_data))

gene = return_models()

# def test_model(net_g,val_data):
#     net_g.eval()
#     x,y = next(iter(val_data))
#     x = x.to(config.device)
#     y = y.to(config.device)
#     for i in range(15):
#         x_i = x[i].unsqueeze(0)
#         y_i = y[i].unsqueeze(0)
#         y_gen = net_g(x_i)
#         image_grid = torch.cat([x_i,y_i,y_gen],dim=3)
#         image_grid = image_grid.permute(0,2,3,1).flatten(start_dim=0,end_dim=1).detach().cpu().numpy()
#         plt.figure(figsize=(10,8))
#         plt.imshow(image_grid)
#         plt.savefig(f'{i+1}_example.svg',format='svg')
#         plt.show()

gene.load_state_dict(torch.load(config.model_path))
gene.eval()

x = torch.randn((1,3,256,256), dtype=torch.float32, device=config.device)

torch.onnx.export(gene,                       # model being run
                  x,                           # model input (or a tuple for multiple inputs)
                  './generator.onnx',             # Path to saved onnx model
                  export_params=True,          # store the trained parameter weights inside the model file
                  opset_version=13,            # the ONNX version to export the model to
                  input_names = ['input'],     # the model's input names
                  output_names = ['output'],   # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})