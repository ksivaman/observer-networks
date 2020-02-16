import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_dataset
from utils import get_features, detect, load_data
from architecture import d1, d2, d3, d4

import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import gzip

import os
import argparse

from models.resnet import ResNet18

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# Load the CIFAR dataset
# Load the MNIST dataset
fp = gzip.open('../../data/cifar/training_data.npy','rb')
x_train = pickle.load(fp)
fp = gzip.open('../../data/cifar/training_labels.npy','rb')
y_train = pickle.load(fp)
fp = gzip.open('../../data/cifar/testing_data.npy','rb')
x_test = pickle.load(fp)
fp = gzip.open('../../data/cifar/testing_labels.npy','rb')
y_test = pickle.load(fp)

# Obtain the model object
model = ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Initialize the classifier
cifar_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer, 
                                     input_shape=(3, 32, 32), nb_classes=10)

# Train the classifier
#cifar_classifier.fit(x_train, y_train, batch_size=128, nb_epochs=10)
state = load_data('../../data/cifar/cifar_target_classifier.npy')
cifar_classifier.__setstate__(state)

# Test the classifier
predictions = cifar_classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy before attack: {}%'.format(accuracy * 100))

#x_test_adv = load_data('../../data/cifar/fgsm_adversarial.npy')
fp = gzip.open('../../data/cifar/fgsm_adversarial.npy','rb')
x_test_adv = pickle.load(fp)

# Test the classifier on adversarial exmaples
predictions = cifar_classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy after attack: {}%'.format(accuracy * 100))

features_a, features_b, features_c, features_d = get_features(cifar_classifier, x_test_adv)
new_accuracy = detect(features_a, features_b, features_c, features_d, d1, d2, d3, d4, x_test_adv)

if (new_accuracy != -1):
    print('Accuracy of detection of adversarial samples on CIFAR-10 using FGSM is: {}%'.format(new_accuracy * 100))
#cifar_classifier.save('cifar_fgsm_state_dict', 'models')

print('exiting cifar-fast gradient sign method...')
