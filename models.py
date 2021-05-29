import torch
import torch.nn as nn


#model source: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
class cifar_10_CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""

        super(cifar_10_CNN, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer_1 = nn.Sequential(
            nn.Linear(3136, 1024), 
            nn.ReLU(), #this layer is the lid layer
        )

        self.fc_layer_2 = nn.Sequential(
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        """Perform forward."""        
        # 3 conv layers
        x = self.conv_layer(x)        
        # flatten
        x = x.view(x.size(0), -1) 
        # fc layer
        lid_feature = self.fc_layer_1(x)
        x = self.fc_layer_2(lid_feature)

        return x, lid_feature