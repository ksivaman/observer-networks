import torch
from torch import nn
from torchvision import models, datasets, transforms

classifier = models.resnet18(pretrained=True)

d1 = torch.nn.Sequential(*(list(classifier.children())[5:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))

d2 = torch.nn.Sequential(*(list(classifier.children())[6:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))

d3 = torch.nn.Sequential(*(list(classifier.children())[7:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))

d4 = torch.nn.Sequential(*(list(classifier.children())[8:-2]), nn.Linear(in_features=512, out_features=10, bias=True), nn.Linear(in_features=10, out_features=2, bias=True))
