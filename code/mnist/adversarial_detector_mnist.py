import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.append('./../')


from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist

from torchvision import transforms
from torchvision.datasets import MNIST

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, fmap=''):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        fmap = x
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, fmap

device = torch.device('cuda')

x_test = torch.from_numpy(torch.load('tensors/test_imgs_mnist.pt'))
x_test = x_test.type(torch.FloatTensor)
x_test = x_test.to(device)

x_test_adv = torch.from_numpy(torch.load('tensors/test_imgs_mnist_adversarial.pt'))
x_test_adv = x_test_adv.type(torch.FloatTensor)
x_test_adv = x_test_adv.to(device)

x_train = torch.from_numpy(torch.load('tensors/train_imgs_mnist.pt'))
x_train = x_train.type(torch.FloatTensor)
x_train = x_train.to(device)

x_train_adv = torch.from_numpy(torch.load('tensors/train_imgs_mnist_adversarial.pt'))
x_train_adv = x_train_adv.type(torch.FloatTensor)
x_train_adv = x_train_adv.to(device)

y_test = torch.from_numpy(torch.load('tensors/test_labels_mnist.pt'))
y_test = y_test.type(torch.FloatTensor)
y_test = y_test.to(device)

model = Net()
model.load_state_dict(torch.load('models/mnist_fgsm_state_dict.model'))
model.to(device)
model.eval()

'''
Uncomment exactly 1 block at once below to avoid gpu memory errors
'''

################################################

'''
correct = 0
total = 0

predictions, fmaps_x_test = model.forward(x_test)
_, predictions = torch.max(predictions, dim=1)
_, y_test = torch.max(y_test, dim=1)

correct = predictions.eq(y_test.view_as(predictions)).sum().item()
total = len(x_test)

torch.save(fmaps_x_test, 'feature_maps/x_test.pt')
'''

################################################

'''
correct = 0
total = 0

predictions_adv, fmaps_x_test_adv = model.forward(x_test_adv)
_, predictions_adv = torch.max(predictions_adv, dim=1)
_, y_test = torch.max(y_test, dim=1)

correct = predictions_adv.eq(y_test.view_as(predictions_adv)).sum().item()
total = len(x_test_adv)

torch.save(fmaps_x_test_adv, 'feature_maps/x_test_adv.pt')
'''

################################################

'''
predictions, fmaps_x_train = model.forward(x_train)
torch.save(fmaps_x_train, 'feature_maps/x_train.pt')
'''

################################################


predictions_adv, fmaps_x_train_adv = model.forward(x_train_adv)
torch.save(fmaps_x_train_adv, 'feature_maps/x_train_adv.pt')

################################################


