import torch
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn


class UNet(torch.nn.Module):
    def __init__(self, input_channels=3, out_channels=3, n_filters=64, latent_space=100):
        super().__init__()
        # Learnable
        self.latent_space = latent_space
        self.conv1A = torch.nn.Conv2d(input_channels, n_filters, 3)
        self.conv1B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.conv2A = torch.nn.Conv2d(n_filters, 2 * n_filters, 3)
        self.conv2B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv3A = torch.nn.Conv2d(2 * n_filters, 4 * n_filters, 3)
        self.conv3B = torch.nn.Conv2d(4 * n_filters, 4 * n_filters, 3)
        self.conv4A = torch.nn.Conv2d(4 * n_filters, 2 * n_filters, 3)
        self.conv4B = torch.nn.Conv2d(2 * n_filters, 2 * n_filters, 3)
        self.conv5A = torch.nn.Conv2d(2 * n_filters, n_filters, 3)
        self.conv5B = torch.nn.Conv2d(n_filters, n_filters, 3)
        self.convtrans34 = torch.nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, 2, stride=2)
        self.convtrans45 = torch.nn.ConvTranspose2d(2 * n_filters, n_filters, 2, stride=2)

        self.convfinal = torch.nn.Conv2d(n_filters, out_channels, 1)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.tanh = torch.nn.Tanh()

    # def forward(self, x):
    def forward(self, x):
        # reshape noise vector to match input size of first convolutional layer
        # x = x.view(-1, self.latent_space, 1, 1)

        l1 = self.relu(self.conv1B(self.relu(self.conv1A(x))))
        l2 = self.relu(self.conv2B(self.relu(self.conv2A(self.pool(l1)))))
        out = self.relu(self.conv3B(self.relu(self.conv3A(self.pool(l2)))))
        out = torch.cat([self.convtrans34(out), l2[:, :, 4:-4, 4:-4]], dim=1)  # copy & crop

        # out = torch.cat([self.convtrans34(out), l2], dim=1)
        out = self.relu(self.conv4B(self.relu(self.conv4A(out))))
        out = torch.cat([self.convtrans45(out), l1[:, :, 16:-16, 16:-16]], dim=1)
        # out = torch.cat([self.convtrans45(out), l1], dim=1)
        out = self.relu(self.conv5B(self.relu(self.conv5A(out))))

        # Finishing
        out = self.convfinal(out)

        return out


class VGG16Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3, pretrained: bool = True):
        super().__init__()
        self.vgg16 = vgg16(
            weights=VGG16_Weights.DEFAULT if pretrained else None)  # loading pretrained VGG16_Weights.DEFAULT
        self.vgg16.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 1),
            torch.nn.Sigmoid()
        )
        self.vgg16.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.vgg16(x)
        return out
