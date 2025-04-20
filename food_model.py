import torch
import torch.nn as nn
import torch.nn.functional as F

class FoodModel(nn.Module):
    def __init__(self, num_classes):
        super(FoodModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        # The input size is 256x256, so after 4 max pooling layers,
        # the output size will be 16x16
        self.fc1 = nn.Linear(16*16*512, 512)
        # self.bn5 = nn.BatchNorm1d(2048)
        # self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        # x = F.relu(self.bn5(self.fc1(x)))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            x = self.forward(x)
            _, predicted = torch.max(x, 1)
        return predicted

    def name(self):
        return self.__class__.__name__