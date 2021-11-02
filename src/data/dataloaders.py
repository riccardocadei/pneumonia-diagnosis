import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.datasets import GMC_Dataset

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
                plt.title(labels[0])
                plt.show()
class GMC_DataLoader(SuperDataLoader):
    def __init__(self, stage = "train", *args, **kwargs):
        self.stage = stage
        self.dataset = GMC_Dataset(data_dir=f'./data/raw/GMC/{stage}/'.format(stage = stage),
                                    image_list_file=f'./data/external/GMC/output_{stage}.txt'.format(stage=stage),
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), #crop each img to the same size for batch
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5832],
                                [0.1412]),
                                        #transforms.TenCrop(224), #for data augmentation crop one img into 10 imgs
                                        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
        super().__init__(dataset = self.dataset, *args, **kwargs)

class NIH_DataLoader(SuperDataLoader):
    def __init__(self, stage = "train", *args, **kwargs):
        self.stage = stage
        self.dataset = GMC_Dataset(data_dir=f'./data/raw/NIH/{stage}/'.format(stage = stage),
                                    image_list_file=f'./data/external/NIH/output_{stage}.txt'.format(stage=stage),
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), #crop each img to the same size for batch
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5362],
                                [0.2001]),
                                        #transforms.TenCrop(224), #for data augmentation crop one img into 10 imgs
                                        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
        super().__init__(dataset = self.dataset, *args, **kwargs)