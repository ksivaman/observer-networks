import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist
from torch.autograd import Variable

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

class MapsDataset(Dataset):
    def __init__(self, maps_path, maps_path_adv, device):
        super(MapsDataset, self).__init__()
        self.maps = torch.load(maps_path)
        self.labels = [0] * self.maps.shape[0] + [1] * self.maps.shape[0]

        self.maps_adv = torch.load(maps_path_adv)

        self.maps = self.maps.type(torch.FloatTensor)
        self.maps_adv = self.maps_adv.type(torch.FloatTensor)

        self.maps = torch.cat((self.maps, self.maps_adv), 0)    
        
    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, idx):
        return self.maps[idx], self.labels[idx]

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv2 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2*2*50, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

device = torch.device('cuda')

model = Detector()
model.to(device)

epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

maps_path = 'tensors/shuffled_train_maps.pt'
labels_path = 'tensors/shuffled_train_labels.pt'
test_maps_path = 'tensors/shuffled_test_maps.pt'
test_labels_path = 'tensors/shuffled_test_labels.pt'

maps = torch.load(maps_path)
labels = torch.load(labels_path)
test_maps = torch.load(test_maps_path)
test_labels = torch.load(test_labels_path)

maps = torch.FloatTensor(maps)
labels = torch.LongTensor(labels)
test_maps = torch.FloatTensor(test_maps)
test_labels = torch.LongTensor(test_labels)

test_maps = test_maps.to(device)
test_labels = test_labels.to(device)
maps = maps.to(device)
labels = labels.to(device)

print((maps.shape), (labels.shape), (test_maps.shape), (test_labels.shape))


start = time.time()
for epoch in range(epochs):

    model.train()
    running_loss = 0.0
  
    for i in range(600):

        print('Batch {} of epoch {}'.format(i, epoch+1))
        maps_tmp = maps[i:i+100, :, :, :]
        labels_tmp = labels[i:i+100]

        maps_tmp, labels_tmp = maps_tmp.to(device), labels_tmp.to(device)
        
        optimizer.zero_grad()

        outputs = model(maps_tmp)
        #_, outputs = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels_tmp)

        print(outputs.shape, labels_tmp.shape)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Loss for epoch {} is {}'.format(epoch+1, running_loss))
    
    model.eval()
    with torch.no_grad():
        test_out = model(test_maps)
        _, test_out = torch.max(test_out, dim=1)
        correct = test_out.eq(test_labels.view_as(test_out)).sum().item()
        total = len(test_labels)
        print('Test accuracy after epoch {} is: {}'.format(epoch+1, correct / total))

end = time.time()

print(end - start)
print('Finished Training')
