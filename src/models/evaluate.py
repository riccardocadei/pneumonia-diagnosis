import torch
from tqdm.notebook import tqdm
import numpy as np


def test(model, data_loader, name_model, device, exp2 = False):
    # load the model
    model.to(device)
    model.load_state_dict(torch.load(f"./models/{name_model}.pth"))
    # test-the-model
    model.eval()  # it-disables-dropout
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device).cpu()
            outputs = model(images)
            predicted = np.around(outputs.cpu())
            predicted, labels = predicted.flatten(), labels.flatten()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Accuracy of the model: {} %'.format(100 * correct / total))
    return 