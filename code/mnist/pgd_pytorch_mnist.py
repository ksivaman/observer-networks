import time
import sys
import os
import torch
import gzip
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

sys.path.append('./../')

from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist
from utils import get_features, detect_pgd, load_obj, load_data
from architecture import d1, d2, d3, d4, Net

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

# Load the MNIST dataset
fp = gzip.open('../../data/mnist/training_data.npy','rb')
x_train = pickle.load(fp)
fp = gzip.open('../../data/mnist/training_labels.npy','rb')
y_train = pickle.load(fp)
fp = gzip.open('../../data/mnist/testing_data.npy','rb')
x_test = pickle.load(fp)
fp = gzip.open('../../data/mnist/testing_labels.npy','rb')
y_test = pickle.load(fp)#set cuda
device = torch.device('cuda')

# Obtain the model object
model = Net()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Initialize the classifier
mnist_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer, 
                                     input_shape=(1, 28, 28), nb_classes=10)

# Train the classifier
#mnist_classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)
state = load_data('../../data/mnist/objects/mnist_target_classifier.pkl')
mnist_classifier.__setstate__(state)

# Test the classifier
predictions = mnist_classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy before attack: {}%'.format(accuracy * 100))

start = time.time()
# Craft the adversarial examples
x_test_adv = load_data('../../data/mnist/pgd_adversarial.npy')

# Test the classifier on adversarial exmaples
predictions = mnist_classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy after attack: {}%'.format(accuracy * 100))

features_a, features_b, features_c, features_d = get_features(mnist_classifier, x_test_adv)
new_accuracy = detect_pgd(features_a, features_b, features_c, features_d, d1, d2, d3, d4, x_test_adv)

if (new_accuracy != -1):
    print('Accuracy after defense on mnist on pgd is: {}%'.format(new_accuracy * 100))
mnist_classifier.save('mnist_pgd_state_dict', 'models')

print('exiting projected gradient descent...')
