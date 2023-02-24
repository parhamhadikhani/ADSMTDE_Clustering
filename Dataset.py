# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:32:43 2021

@author: User
"""
import torch
from torch import nn
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
#from torchvision.utils import save_image
import os

import numpy as np
import pandas as pd

def load_mnist():
	train = MNIST(root='./data/',
	            train=True, 
	            transform=transforms.ToTensor(),
	            download=True)

	test = MNIST(root='./data/',
	            train=False, 
	            transform=transforms.ToTensor())
	x_train, y_train = train.train_data, train.train_labels
	x_test, y_test = test.test_data, test.test_labels
	x = torch.cat((x_train, x_test), 0)
	y = torch.cat((y_train, y_test), 0)
	x = x.reshape((x.shape[0], -1))
	x = np.divide(x, 255.)
	print('MNIST samples', x.shape)
	y_names=['0','1','2','3','4','5','6','7','8','9']

	return x, y,y_names

def load_har():
    x_train = pd.read_csv(
        'data/har/train/X_train.txt',
        sep=r'\s+',
        header=None)
    y_train = pd.read_csv('data/har/train/y_train.txt', header=None)
    x_test = pd.read_csv('data/har/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv('data/har/test/y_test.txt', header=None)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # labels start at 1 so..
    y = y - 1
    y = y.reshape((y.size,))
    y_names = ['Walking','Upstairs','Downstairs','Sitting','Standing','Laying']
    return x, y, y_names


def load_usps(data_path='data/usps'):
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    y_names=['0','1','2','3','4','5','6','7','8','9']

    return x, y,y_names


def load_pendigits(data_path='data/pendigits'):
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.makedirs(data_path,  exist_ok=True)
        
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra -P %s' %
            data_path)
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes -P %s' %
            data_path)
        os.system(
                'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names -P %s' %
            data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    y = y.astype('int')
    y_names=['0','1','2','3','4','5','6','7','8','9']

    return x, y,y_names


def load_fashion():
    y_names = ['T-shirt','Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle Boot']
    #D:\PG\MTDE\MTDE\fashion
    x=np.load(r'D:/PG/MTDE/MTDE/fashion/DATA.npy')
    y=np.load(r'D:/PG/MTDE/MTDE/fashion/LABEL.npy')
    return x,y,y_names
