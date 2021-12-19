import torch
from src.models.resnet import get_resnet
from src.models.style_encoder import ConvEncoder


class SuperModel(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SuperModel, self).__init__()
        self.image_encoder = get_resnet(50, 2, 0.0625)
        self.image_encoder.load_state_dict(torch.load("models/r50_2x_sk1.pth")['resnet'])
        self.style_encoder =  ConvEncoder()
        self.linear1 = torch.nn.Linear(4096+1024, 2048)
        self.linear2 = torch.nn.Linear(2048,512)
        self.linear3 = torch.nn.Linear(512,128)
        self.linear4 = torch.nn.Linear(128,32)
        self.linear5 = torch.nn.Linear(32,1)
        self.softmax = torch.nn.Sigmoid()
        self.name = "supermodel"

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x_ray = x[:1]
        style_batch = x[-4:]

        self.image_encoder.eval()
        with torch.no_grad():
            x_latent = self.image_encoder(x_ray)
        self.image_encoder.train()

        style_batch_latent =  torch.unsqueeze(self.style_encoder(style_batch),0)

        X = torch.cat((x_latent[0,None],style_batch_latent),1)

        y_pred = self.linear1(X)
        y_pred = self.linear2(y_pred)
        y_pred = self.linear3(y_pred)
        y_pred = self.linear4(y_pred)
        y_pred = self.linear5(y_pred)
        y_pred = self.softmax(y_pred)

        return y_pred
