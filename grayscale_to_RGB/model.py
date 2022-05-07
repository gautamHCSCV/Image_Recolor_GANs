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
import wandb

# api_key = 'ae12d9032b94bfedc39f2e1beacfbf9909359ffc'
# os.environ['WANDB_API_KEY'] = api_key
class cnnblock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2):
        super(cnnblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                bias=False,
                padding_mode='reflect'
                ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)

class discriminator(nn.Module):
    # 256 -> 26,26
    def __init__(self,in_channels=3,features=[64,128,256,512]):
        super(discriminator,self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels*2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect'
                ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        self.in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                cnnblock(self.in_channels,feature,stride= 1 if feature == features[-1] else 2)
            )
            self.in_channels = feature

        layers.append(
            nn.Conv2d(self.in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode='reflect')
        )
        self.model = nn.Sequential(*layers)        

    def forward(self,x,y): # x , y (fake / real)
        z = torch.cat([x,y],dim=1)
        z = self.initial(z)
        return self.model(z)


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
    disc = discriminator()

    gene = gene.to(config.device)
    disc = disc.to(config.device)

    return gene, disc


def return_optimizer_loss(gene,disc):
    ogen = optim.Adam(gene.parameters(),lr=config.lr,betas=(0.5,0.999))
    odis = optim.Adam(disc.parameters(),lr=config.lr,betas=(0.5,0.999))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()
    return ogen,odis,bce,l1_loss,g_scalar,d_scalar

def test():
    random_img = torch.rand(1,3,256,256,device=config.device)
    random_img1 = torch.rand(1,3,256,256,device=config.device)
    model = generator()
    model = model.to(config.device)
    print(model(random_img).shape)

def train(net_g,net_d,train_data,bce,l1_loss,o_gen,o_dis,g_scalar,d_scalar,n_epochs = config.epochs):
    api_key = 'ae12d9032b94bfedc39f2e1beacfbf9909359ffc'
    os.environ['WANDB_API_KEY'] = api_key
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="DLOPs Project")
    net_g.train()
    net_d.train()
    time1 = time.perf_counter()
    history = {'genloss':[],'discloss':[]}

    for epoch in range(n_epochs):
        wandb.watch(net_g, bce, log="all", log_freq=10)
        wandb.watch(net_g, l1_loss, log="all", log_freq=10)
        wandb.watch(net_d, bce, log="all", log_freq=10)

        running_gene_loss = 0;
        running_disc_loss = 0;
        len_train = len(train_data)
        for idx, (src,tar) in enumerate(train_data):

            src = src.to(config.device)
            tar = tar.to(config.device)

            with torch.cuda.amp.autocast():
                y_fake = net_g(src)
                d_real = net_d(src,tar)
                d_fake = net_d(src,y_fake.detach())
                d_real_loss = bce(d_real,torch.ones_like(d_real))
                d_fake_loss = bce(d_fake,torch.zeros_like(d_fake))
                d_loss = (d_real_loss + d_fake_loss) / 2
            
            net_d.zero_grad()
            d_scalar.scale(d_loss).backward()
            d_scalar.step(o_dis)
            d_scalar.update()

            with torch.cuda.amp.autocast():
                d_fake = net_d(src,y_fake)
                g_fake_loss = bce(d_fake,torch.ones_like(d_fake))
                l1 = l1_loss(y_fake,tar) * 100
                g_loss = (g_fake_loss + l1)
            
            o_gen.zero_grad()
            g_scalar.scale(g_loss).backward()
            g_scalar.step(o_gen)
            g_scalar.update()

            history['genloss'].append(g_loss.item())
            history['discloss'].append(d_loss.item())

            running_gene_loss += (g_loss.item()/len_train)
            running_disc_loss += (d_loss.item()/len_train)
            wandb.log({'gen_loss':g_loss.item()})
            wandb.log({'disc_loss':d_loss.item()})

            if config.verbose:
                print(f'batches_done:{idx}',end='\r')
        
        print(f'epoch: [{epoch+1}/{n_epochs}], gene_loss: {running_gene_loss:.2f}, disc_loss: {running_disc_loss:.2f}')

    time2 = time.perf_counter() - time1
    print(f'training completed in {int(time2 // 60)} Mins {int(time2 % 60)} Sec')
    return history



# test()