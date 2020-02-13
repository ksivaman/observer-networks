import torch
from torch import nn
from torchvision import models, datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

#Create the neural network architecture, return logits instead of activation in forward method (Eg. softmax).
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

classifier = models.resnet18(pretrained=False)

d1 = torch.nn.Sequential(*(list(classifier.children())[5:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))

d2 = torch.nn.Sequential(*(list(classifier.children())[6:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))

d3 = torch.nn.Sequential(*(list(classifier.children())[7:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))

d4 = torch.nn.Sequential(*(list(classifier.children())[8:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))
