#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# arguments define
import argparse

# load torch
import torchvision

# other utilities
# import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix


# %% Load the training data
def MNIST_DATASET_TRAIN(downloads, train_amount):
    # Load dataset
    training_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=downloads
    )

    # Convert Training data to numpy
    train_data = training_data.train_data.numpy()[:train_amount]
    train_label = training_data.train_labels.numpy()[:train_amount]

    # Print training data size
    # print('Training data size: ', train_data.shape)
    # print('Training data label size:', train_label.shape)
    # plt.imshow(train_data[0])
    # plt.show()

    train_data = train_data / 255.0

    return train_data, train_label


# %% Load the test data
def MNIST_DATASET_TEST(downloads, test_amount):
    # Load dataset
    testing_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=downloads
    )

    # Convert Testing data to numpy
    test_data = testing_data.test_data.numpy()[:test_amount]
    test_label = testing_data.test_labels.numpy()[:test_amount]

    # Print training data size
    # print('test data size: ', test_data.shape)
    # print('test data label size:', test_label.shape)
    # plt.imshow(test_data[0])
    # plt.show()

    test_data = test_data / 255.0

    return test_data, test_label


# Training Arguments Settings
parser = argparse.ArgumentParser(description='Saak')
parser.add_argument('--download_MNIST', default=True, metavar='DL',
                    help='Download MNIST (default: True)')
parser.add_argument('--train_amount', type=int, default=60000,
                    help='Amount of training samples')
parser.add_argument('--test_amount', type=int, default=2000,
                    help='Amount of testing samples')
args = parser.parse_args()

# Load Training Data & Testing Data
train_data, train_label = MNIST_DATASET_TRAIN(args.download_MNIST, args.train_amount)
test_data, test_label = MNIST_DATASET_TEST(args.download_MNIST, args.test_amount)

training_features = train_data.reshape(args.train_amount, -1)
test_features = test_data.reshape(args.test_amount, -1)

# Training SVM
clf = svm.SVC(C=5, gamma=0.05, max_iter=100)
clf.fit(training_features, train_label)

# Train data
train_result = clf.predict(training_features)
precision = sum(train_result == train_label) / train_label.shape[0]
print('训练集的精确度为: ', precision)

# Test data
test_result = clf.predict(test_features)
precision = sum(test_result == test_label) / test_label.shape[0]
print('测试集的精确度为: ', precision)


# Show the confusion matrix
matrix = confusion_matrix(test_label, test_result)
print(matrix)