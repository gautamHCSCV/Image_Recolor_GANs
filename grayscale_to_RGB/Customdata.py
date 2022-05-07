from PIL import Image
import torch
import torchvision
import numpy as np
import os
from config import config
import matplotlib.pyplot as plt


class stl10data(object):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize((config.image_size,config.image_size)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.root_dir,image_name)
        img = Image.open(image_path)
        input_ig = img.convert('L')
        output_ig = img.convert('RGB')
        input_img = self.transformations(input_ig)
        input_img = torch.cat([input_img,input_img,input_img],dim=0)
        output_img = self.transformations(output_ig)
        return (input_img,output_img)


def get_dataloader():
    train_map_data = stl10data(config.train_path)
    test_map_data = stl10data(config.val_path)

    train_loader = torch.utils.data.DataLoader(
        train_map_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory = config.pin_memory,
        num_workers=config.num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        test_map_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers
    )

    return train_loader, val_loader


# load_img('1005.jpg')
if __name__ == '__main__':
    pass
    # train_loader, test_loader = get_dataloader()
    # img1,img2 = next(iter(train_loader))
    # a = img1[0].permute(1,2,0).detach().cpu().numpy()
    # print(img1.shape,img2.shape)
    # plt.imshow(a)
    # plt.savefig('a.png')