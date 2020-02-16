import time
import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle


sys.path.append('./../')

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist
from utils import get_features, detect, load_obj, load_data, save_data
from architecture import d1, d2, d3, d4, Net
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

# Load the MNIST dataset
x_train = load_data('../../data/mnist/train_data.npy')
y_train = load_data('../../data/mnist/train_labels.npy')
x_test = load_data('../../data/mnist/test_data.npy')
y_test = load_data('../../data/mnist/test_labels.npy')

# Obtain the model object
model = Net()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Initialize the classifier
mnist_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer, 
                                     input_shape=(1, 28, 28), nb_classes=10)


state = load_data('../../data/mnist/objects/mnist_target_classifier.pkl')
mnist_classifier.__setstate__(state)

# Test the classifier
predictions = mnist_classifier.predict(x_test)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy before attack: {}%'.format(accuracy * 100))

start = time.time()
# Craft the adversarial examples
epsilon = 0.2  # Maximum perturbation
adv_crafter = FastGradientMethod(mnist_classifier, eps=epsilon)
x_test_adv = adv_crafter.generate(x=x_test)
save_data(x_test_adv, '../../data/mnist/fgsm_adversarial.npy')


