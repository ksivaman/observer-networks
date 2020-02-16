import time
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.append('./../')

from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist
from utils import get_features, detect_pgd, load_obj, load_data, save_data
from architecture import d1, d2, d3, d4, Net

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

# Load the MNIST dataset
x_train = load_data('../../data/mnist/train_data.npy')
y_train = load_data('../../data/mnist/train_labels.npy')
x_test = load_data('../../data/mnist/test_data.npy')
y_test = load_data('../../data/mnist/test_labels.npy')

#set cuda
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
epsilon = 0.2  # Maximum perturbation
adv_crafter = ProjectedGradientDescent(mnist_classifier)
x_test_adv = adv_crafter.generate(x=x_test)
save_data(x_test_adv, '../../data/mnist/pgd_adversarial.npy')

