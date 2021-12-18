import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class Super_Dataset(Dataset):
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        label = self.labels[index]

        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

class ENV_Dataset(Super_Dataset):
    def __init__(self, data_dir, stage, image_list_file, transform=None, limit = None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        image_names = []
        labels = []
        datasets = []
        with open(image_list_file, "r") as f:
            for num, line in enumerate(f):
                if limit:
                    if num >= limit:
                        break
                items = line.split()
                dataset = items[0]
                image_name= items[1]
                label = items[2]
                label = [int(i) for i in label]
                if label[0] == 0:
                    image_name = os.path.join('NORMAL/',image_name)
                elif label[0] == 1:
                    image_name = os.path.join('PNEUMONIA/',image_name)

                image_name = os.path.join('{folder}/'.format(folder = stage), image_name)

                if dataset == 'GMC':
                    image_name = os.path.join('GMC/', image_name)
                elif dataset == 'NIH':
                    image_name = os.path.join('NIH/', image_name)

                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform



class GMC_Dataset(Super_Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, limit = None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        # with open(image_list_file, "w+") as a:
        #     for path, _, files in os.walk(data_dir):
        #         for filename in files:
        #             if "DS_Store" not in filename:
        #                 if 'PNEUMONIA' in path:
        #                     label = '1'
        #                 else:
        #                     label = '0'
        #                 a.write(str(filename)+' '+label+os.linesep)


        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for num, line in enumerate(f):
                if limit:
                    if num >= limit:
                        break
                items = line.split()
                image_name= items[0]
                label = items[1]
                label = [int(i) for i in label]
                if label[0] == 0:
                    image_name = os.path.join('NORMAL/',image_name)
                elif label[0] == 1:
                    image_name = os.path.join('PNEUMONIA/',image_name)
                
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

