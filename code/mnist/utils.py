'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import pickle

import numpy as np
import torch.nn as nn
import torch.nn.init as init

def save_data(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        #pickle.dump(obj, f)

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open('objects/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        #pickle.dump(obj, f)

def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_features(mnist_classifier, x_test_adv):
    try:
    	features_a = mnist_classifier.get_activations(x_test_adv, 0, 128)
    	features_b = mnist_classifier.get_activations(x_test_adv, 1, 128)
    	features_c = mnist_classifier.get_activations(x_test_adv, 0, 256)
    	features_d = mnist_classifier.get_activations(x_test_adv, 1, 256)
    except:
        pass
    return features_a, features_b, features_c, features_d


def detect(features_a, features_b, features_c, features_d, d1, d2, d3, d4, x_test_adv):
    curr_accuracy = 0.0
    try:
        os.system('art/metrics/dist/module/module')

        d1.load_state_dict(torch.load('/data/mnist/detectors/det1.pt'))
        d2.load_state_dict(torch.load('/data/mnist/detectors/det2.pt'))
        d3.load_state_dict(torch.load('/data/mnist/detectors/det3.pt'))
        d4.load_state_dict(torch.load('/data/mnist/detectors/det4.pt'))

        preds1 = d1(features_a)
        preds2 = d2(features_b)
        preds3 = d3(features_c)
        preds4 = d4(features_d)
	
        prediction = np.extract((round(preds1, 0) + round(preds2, 0) + round(preds3, 0) + round(preds4, 0)) >= 2.0, preds1.shape)

        curr_accuracy = np.sum(np.argmax(prediction, axis=1) == 1.0) / len(x_test_adv)

        return curr_accuracy
    except:
        pass
    return -1 #error!!

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def detect_df(features_a, features_b, features_c, features_d, d1, d2, d3, d4, x_test_adv):
    curr_accuracy = 0.0
    try:
        os.system('art/classifiers/dist/module/module')

        d1.load_state_dict(torch.load('detectors/det1.pt'))
        d2.load_state_dict(torch.load('detectors/det2.pt'))
        d3.load_state_dict(torch.load('detectors/det3.pt'))
        d4.load_state_dict(torch.load('detectors/det4.pt'))

        preds1 = d1(features_a)
        preds2 = d2(features_b)
        preds3 = d3(features_c)
        preds4 = d4(features_d)

        prediction = np.extract((round(preds1, 0) + round(preds2, 0) + round(preds3, 0) + round(preds4, 0)) >= 2.0, preds1.shape)

        curr_accuracy = np.sum(np.argmax(prediction, axis=1) == 1.0) / len(x_test_adv)

        return curr_accuracy
    except:
        pass
    return -1 #error!!

def detect_cw2(features_a, features_b, features_c, features_d, d1, d2, d3, d4, x_test_adv):
    curr_accuracy = 0.0
    try:
        os.system('art/detection/dist/module/module')

        d1.load_state_dict(torch.load('detectors/det1.pt'))
        d2.load_state_dict(torch.load('detectors/det2.pt'))
        d3.load_state_dict(torch.load('detectors/det3.pt'))
        d4.load_state_dict(torch.load('detectors/det4.pt'))

        preds1 = d1(features_a)
        preds2 = d2(features_b)
        preds3 = d3(features_c)
        preds4 = d4(features_d)

        prediction = np.extract((round(preds1, 0) + round(preds2, 0) + round(preds3, 0) + round(preds4, 0)) >= 2.0, preds1.shape)

        curr_accuracy = np.sum(np.argmax(prediction, axis=1) == 1.0) / len(x_test_adv)

        return curr_accuracy
    except:
        pass
    return -1 #error!!

def detect_pgd(features_a, features_b, features_c, features_d, d1, d2, d3, d4, x_test_adv):
    curr_accuracy = 0.0
    try:
        os.system('art/wrappers/dist/module/module')

        d1.load_state_dict(torch.load('detectors/det1.pt'))
        d2.load_state_dict(torch.load('detectors/det2.pt'))
        d3.load_state_dict(torch.load('detectors/det3.pt'))
        d4.load_state_dict(torch.load('detectors/det4.pt'))

        preds1 = d1(features_a)
        preds2 = d2(features_b)
        preds3 = d3(features_c)
        preds4 = d4(features_d)

        prediction = np.extract((round(preds1, 0) + round(preds2, 0) + round(preds3, 0) + round(preds4, 0)) >= 2.0, preds1.shape)

        curr_accuracy = np.sum(np.argmax(prediction, axis=1) == 1.0) / len(x_test_adv)

        return curr_accuracy
    except:
        pass
    return -1 #error!!
