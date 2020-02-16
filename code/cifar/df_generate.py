import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_dataset
from utils import get_features, detect, load_data, save_data
from architecture import d1, d2, d3, d4

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models.resnet import ResNet18

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the CIFAR dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)

# Initialize the classifier
cifar_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer, 
                                     input_shape=(3, 32, 32), nb_classes=10)

# Train the classifier
#cifar_classifier.fit(x_train, y_train, batch_size=128, nb_epochs=10)
state = load_data('../../data/cifar/cifar_target_classifier.npy')
cifar_classifier.__setstate__(state)

# Craft the adversarial examples
epsilon = 0.4  # Maximum perturbation
adv_crafter = FastGradientMethod(cifar_classifier, eps=epsilon)
x_test_adv = adv_crafter.generate(x=x_test)
save_data(x_test_adv, '../../data/cifar/df_adversarial.npy')
