"""
Main module to define all the block classes for the model.
"""
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Main class to define a basic convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layer(x)


class FCBlock(nn.Module):

    def __init__(self, in_features, out_features):
        """
        Main class to define a basic fully connected block.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layer(x)


class Classifier(nn.Module):

    def __init__(self, img_size, image_channels, feature_scale = 8, n_classes = 10):
        """
        Main class to define the classifier model.

        Args:
            img_size (int): Size of the input image.
            image_channels (int): Number of image channels.
            feature_scale (int): Number of feature scales.
            n_classes (int): Number of classes.
        """
        super().__init__()

        # Define the convolutional blocks
        self.conv1 = ConvBlock(in_channels=image_channels, out_channels=feature_scale * 1)
        self.conv2 = ConvBlock(in_channels=feature_scale * 1, out_channels=feature_scale * 2)
        self.conv3 = ConvBlock(in_channels=feature_scale * 2, out_channels=feature_scale * 4)

        # Define the fully connected block
        img_featutes = (img_size // 8) ** 2 * feature_scale * 4
        self.fc = FCBlock(in_features=img_featutes, out_features=n_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
