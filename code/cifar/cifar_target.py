import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_dataset
from utils import get_features, detect, save_data
from architecture import d1, d2, d3, d4

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models.resnet import ResNet18

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

# Load the CIFAR dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)

save_data(x_train, '../../data/cifar/training_data.npy')
save_data(y_train, '../../data/cifar/training_labels.npy')
save_data(x_test, '../../data/cifar/testing_data.npy')
save_data(y_test, '../../data/cifar/testing_labels.npy')

# Obtain the model object
model = ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Initialize the classifier
cifar_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer, 
                                     input_shape=(3, 32, 32), nb_classes=10)

# Train the classifier
#cifar_classifier.fit(x_train, y_train, batch_size=128, nb_epochs=10)


#state = cifar_classifier.__getstate__()
#save_data(state, '../../data/cifar/cifar_target_classifier.npy')


