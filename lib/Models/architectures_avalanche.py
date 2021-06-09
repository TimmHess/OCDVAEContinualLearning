from collections import OrderedDict
import torch
import torch.nn as nn

from avalanche.models.dynamic_modules import IncrementalClassifier


def get_feat_size(block, spatial_size, ncolors=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.

    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    """

    x = torch.randn(2, ncolors, spatial_size, spatial_size)
    out = block(x)
    num_feat = out.size(1)
    spatial_dim_x = out.size(2)
    spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y

def set_previous_mu_and_std(model, prev_mu, prev_std):
    model.prev_mu = prev_mu
    model.prev_std = prev_std
    return


class DCNNNoVAE(nn.Module):
    def __init__(self, num_classes):
        super(DCNNNoVAE, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.features, 64, 3)

        self.classifier = nn.Sequential(
            nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels, num_classes, bias=False)
        )
        return

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DCNNNoVAEIncremental(nn.Module):
    def __init__(self, num_classes):
        #super(DCNNNoVAEIncremental, self).__init__(in_features=3, initial_out_features=num_classes)
        super(DCNNNoVAEIncremental, self).__init__()
        self.in_features = 3
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.features, 64, 3)

        self.classifier = IncrementalClassifier(in_features=self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                            initial_out_features=2) # TODO: fixme

        #self.classifier = nn.Sequential(
        #    nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels, num_classes, bias=False)
        #)
        return

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
