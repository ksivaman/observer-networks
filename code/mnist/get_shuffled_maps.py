import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys

sys.path.append('./../')

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

def shuffle(tens):
    test_inds = torch.randperm(tens.shape[0])
    m = 0
 
    for num_iter, i in enumerate(test_inds):
        print('set num {}'.format(num_iter))
        if num_iter != 0:
            m = torch.cat((tens[i].unsqueeze(0), m), 0)
        else:
            m = tens[i].unsqueeze(0)

    return m

device = torch.device('cuda')

'''
train_maps = torch.load('feature_maps/x_train.pt')
train_maps = train_maps.type(torch.FloatTensor)
train_maps = train_maps.to(device)

test_maps = torch.load('feature_maps/x_test.pt')
test_maps = test_maps.type(torch.FloatTensor)
test_maps = test_maps.to(device)

train_maps_adv = torch.load('feature_maps/x_train_adv.pt')
train_maps_adv = train_maps_adv.type(torch.FloatTensor)
train_maps_adv = train_maps_adv.to(device)

test_maps_adv = torch.load('feature_maps/x_test_adv.pt')
test_maps_adv = test_maps_adv.type(torch.FloatTensor)
test_maps_adv = test_maps_adv.to(device)
'''

model = Detector()
model.to(device)

epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_set = MapsDataset('feature_maps/x_train.pt', 'feature_maps/x_train_adv.pt', device)
test_set = MapsDataset('feature_maps/x_test.pt', 'feature_maps/x_test_adv.pt', device)

trainloader  = DataLoader(dataset=train_set, 
                         shuffle=False, 
                         batch_size=64, 
                         num_workers=1)

testloader  = DataLoader(dataset=test_set,
                         shuffle=True,
                         batch_size=64,
                         num_workers=1)
'''
model.train()
for epoch in range(epochs):

    running_loss = 0.0
    for inputs, labels in trainloader:
        
        #inputs = Variable(inputs, requires_grad=True)
        #labels = Variable(labels, requires_grad=True)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Loss for epoch {} is {}'.format(epoch, running_loss))    
'''

maps_path = 'feature_maps/x_train.pt'
maps_path_adv = 'feature_maps/x_train_adv.pt'

maps = torch.load(maps_path)

labels = [0] * maps.shape[0] + [1] * maps.shape[0]
labels = torch.LongTensor(labels)

maps_adv = torch.load(maps_path_adv)
maps = maps.type(torch.FloatTensor)
maps_adv = maps_adv.type(torch.FloatTensor)

maps = torch.cat((maps, maps_adv), 0)
inds = torch.randperm(maps.shape[0])


print(maps.shape)
print(labels.shape)

#maps = maps[inds, :, :, :]
#labels = labels[inds]
maps = shuffle(maps)

print(maps.shape)
print(labels.shape)

test_maps_path = 'feature_maps/x_test.pt'
test_maps_path_adv = 'feature_maps/x_test_adv.pt'

test_maps = torch.load(test_maps_path)

test_labels = [0] * test_maps.shape[0] + [1] * test_maps.shape[0]
test_labels = torch.LongTensor(test_labels)

test_maps_adv = torch.load(test_maps_path_adv)
test_maps = test_maps.type(torch.FloatTensor)
test_maps_adv = test_maps_adv.type(torch.FloatTensor)

test_maps = torch.cat((test_maps, test_maps_adv), 0)
test_inds = torch.randperm(test_maps.shape[0])
'''test_maps = test_maps[test_inds, :, :, :]
test_labels = test_labels[test_inds]'''

test_maps = shuffle(test_maps)

torch.save(maps, 'tensors/shuffled_train_maps.pt')
torch.save(test_maps, 'tensors/shuffled_test_maps.pt')

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
        loss = criterion(outputs, labels_tmp)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Loss for epoch {} is {}'.format(epoch+1, running_loss))
    
    model.eval()
    with torch.no_grad():
        test_out = model(test_maps)
        correct = test_out.eq(test_labels.view_as(test_out)).sum().item()
        total = len(test_labels)
        print('Test accuracy after epoch {} is: {}'.format(epoch+1, correct / total))
    
print('Finished Training')
