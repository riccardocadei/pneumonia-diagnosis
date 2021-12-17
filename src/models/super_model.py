import torch
from src.models.resnet import get_resnet
from src.models.style_encoder import ConvEncoder


class SuperModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SuperModel, self).__init__()
        self.image_encoder = get_resnet(50, 2, 0.0625)
        self.style_encoder =  ConvEncoder()
        #self.linear1 = torch.nn.Linear(4096+style_encoder_out, l1)
        #self.linear2 = torch.nn.Linear(l1,l2)
        #self.linear3 = torch.nn.Linear(l2,l3)
        #self.softmax = torch.nn.Softmax()
    def forward(self, xray, style_batch):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        x_latent = self.image_encoder(xray)
        style_batch_latent = self.style_encoder(style_batch)

        y_pred = self.linear1([x_latent,style_batch_latent])
        y_pred = self.linear2(y_pred)
        y_pred = self.linear3(y_pred)
        y_pred = self.softmax(y_pred)

        return y_pred
