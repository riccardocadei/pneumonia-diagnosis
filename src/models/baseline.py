import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    """Baseline model reproduced from arXiv:2007.10653 and indentend to compare in the first experiment 
    to SOTA model in robustness evaluation.

    """
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3, padding=1)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(64,128,3, padding=1)
        self.elu2 = nn.ELU()
        self.outconv = nn.Conv2d(128,1,3,padding=1)
        self.elu3 = nn.ELU()
        self.final_linear1 = nn.Linear(224,1)
        self.final_linear2 = nn.Linear(224,1)
        self.sigmoid = nn.Sigmoid()

        self.name = "Baseline"

    def forward(self,x):
        x = self.conv1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.elu2(x)
        x = self.outconv(x)
        x = self.elu3(x)
        x = self.final_linear1(x)
        x = self.final_linear2(x.squeeze())
        x = self.sigmoid(x)

        return x