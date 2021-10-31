import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from readdata import ChestXrayDataSet
import matplotlib.pyplot as plt

if __name__ == '__main__':


    DATA_DIR = './chest_xray/val/NORMAL'
    TEST_IMAGE_LIST = './output.txt'
    BATCH_SIZE = 4

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                        image_list_file=TEST_IMAGE_LIST,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224), #crop each img to the same size for batch
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: normalize(x))
                                            #transforms.TenCrop(224), #for data augmentation crop one img into 10 imgs
                                            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0, pin_memory=True)


#dataloader vizualization
for batch_idx, (inputs, labels) in enumerate(test_loader):
    inputs = inputs[0,:,:,:]
    print(inputs.shape)
    inputs = inputs.permute(1, 2, 0)
    plt.figure()
    plt.imshow(inputs.numpy())
    plt.show()
