import time
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

sys.path.append('./../')

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist
from utils import save_obj, save_data
from architecture import d1, d2, d3, d4, Net

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)

#save_data(x_train, '../../data/mnist/train_data.npy')
#save_data(y_train, '../../data/mnist/train_labels.npy')
#save_data(x_test, '../../data/mnist/test_data.npy')
#save_data(y_test, '../../data/mnist/test_labels.npy')

#set cuda
#device = torch.device('cuda')

# Obtain the model object
model = Net()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Initialize the classifier
mnist_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer, 
                                     input_shape=(1, 28, 28), nb_classes=10)

# Train the classifier
mnist_classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)

state = mnist_classifier.__getstate__()

for k, v in state.items():
    print(k, v)

save_data(state, '../../data/mnist/objects/mnist_target_classifier.npy')

