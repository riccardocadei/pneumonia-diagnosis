import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.data.datasets import GMC_Dataset
from src.data.datasets import ENV_Dataset

class SuperDataLoader(DataLoader):
    def show(self, iter = 10, type = None):
        num = 0
        for _, (inputs, labels) in enumerate(self):
            if num == iter:
                break

            show = False

            if type == 'PNEUMONIA' and labels[0] == 1:
                show = True
            elif type == 'NORMAL' and labels[0] == 0:
                show = True
            elif type == None:
                show = True
            if show:
                num +=1 
                inputs = inputs[0,:,:]
                inputs = inputs.permute(1, 2, 0)
                plt.figure()
                plt.imshow(inputs.numpy())
                plt.title('Pneumonia' if labels[0] else 'Normal')
                plt.show()


class ENV_DataLoader(SuperDataLoader):
    def __init__(self, stage = "train", type = 0, limit = None, *args, **kwargs):
        self.stage = stage
        image_list_file = ""

        if stage == 'train' and type == 1:
            image_list_file = './data/labels/ENV/output_1_train.txt'
        elif stage == 'train' and type == 2:
            image_list_file = './data/labels/ENV/output_2_train.txt'
        elif stage == 'test':
            image_list_file = './data/labels/ENV/output_0_test.txt'
        else:
            print('Error')



        self.dataset = ENV_Dataset(data_dir=f'./data/images/'.format(stage = stage),
                                    image_list_file=image_list_file,
                                    stage = self.stage,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), #crop each img to the same size for batch
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5832,0.5832,0.5832],
                                [0.1412,0.1412,0.1412]),
                                        #transforms.TenCrop(224), #for data augmentation crop one img into 10 imgs
                                        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]), limit = limit)
        super().__init__(dataset = self.dataset, *args, **kwargs)


class GMC_DataLoader(SuperDataLoader):
    def __init__(self, stage = "train", limit = None, balanced = False, *args, **kwargs):
        self.stage = stage
        if balanced:
            self.list_file = f'./data/labels/GMC/balanced_output_{stage}.txt'.format(stage=stage)
        else:
            self.list_file = f'./data/labels/GMC/output_{stage}.txt'.format(stage=stage)

        self.dataset = GMC_Dataset(data_dir=f'./data/images/GMC/{stage}/'.format(stage = stage),
                                    image_list_file=self.list_file,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), #crop each img to the same size for batch
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5832],
                                [0.1412]),
                                        #transforms.TenCrop(224), #for data augmentation crop one img into 10 imgs
                                        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]), limit = limit)
        super().__init__(dataset = self.dataset, *args, **kwargs)

class NIH_DataLoader(SuperDataLoader):
    def __init__(self, stage = "train", limit = None, balanced = False, *args, **kwargs):
        self.stage = stage
        if balanced:
            self.list_file = f'./data/labels/NIH/balanced_output_{stage}.txt'.format(stage=stage)
        else:
            self.list_file = f'./data/labels/NIH/output_{stage}.txt'.format(stage=stage)

        self.dataset = GMC_Dataset(data_dir=f'./data/images/NIH/{stage}/'.format(stage = stage),
                                    image_list_file=self.list_file,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), #crop each img to the same size for batch
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5362,0.5362,0.5362],
                                [0.2001,0.2001,0.2001]),
                                        #transforms.TenCrop(224), #for data augmentation crop one img into 10 imgs
                                        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]), limit = limit)
        super().__init__(dataset = self.dataset, *args, **kwargs)