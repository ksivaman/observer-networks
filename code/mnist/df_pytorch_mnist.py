import time
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

sys.path.append('./../')

from art.attacks.deepfool import DeepFool
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist
from utils import get_features, detect_df
from architecture import d1, d2, d3, d4, Net

cmd = 'art/dist/init_module/init_module'
os.system(cmd)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)

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
mnist_classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)

# Test the classifier
predictions = mnist_classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy before attack: {}%'.format(accuracy * 100))

start = time.time()
# Craft the adversarial examples
epsilon = 0.2  # Maximum perturbation
adv_crafter = DeepFool(mnist_classifier)
x_test_adv = adv_crafter.generate(x=x_test)
# x_train_adv = adv_crafter.generate(x=x_train)

end = time.time()

# Test the classifier on adversarial exmaples
predictions = mnist_classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy after attack: {}%'.format(accuracy * 100))

features_a, features_b, features_c, features_d = get_features(mnist_classifier, x_test_adv)
new_accuracy = detect_df(features_a, features_b, features_c, features_d, d1, d2, d3, d4, x_test_adv)

if (new_accuracy != -1):
    print('Accuracy after defense on mnist on df is: {}%'.format(new_accuracy * 100))
mnist_classifier.save('mnist_df_state_dict', 'models')

print('exiting deep-fool...')
