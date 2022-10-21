import os
import torch
import numpy as np
import datasets


def get_data(path):
    with open(os.path.join(path, 'train_x.npy'), 'rb') as f:
        train_x = np.load(f)

    with open(os.path.join(path, 'test_x.npy'), 'rb') as f:
        test_x = np.load(f)

    with open(os.path.join(path, 'train_y.npy'), 'rb') as f:
        train_y = np.load(f)

    with open(os.path.join(path, 'test_y.npy'), 'rb') as f:
        test_y = np.load(f)

    return train_x, train_y, test_x, test_y


def get_tensors(dataset):

    train_x, train_y, test_x, test_y = dataset

    train_x = torch.from_numpy(train_x).cuda()
    train_y = torch.from_numpy(train_y).cuda()
    test_x = torch.from_numpy(test_x).cuda()
    test_y = torch.from_numpy(test_y).cuda()
    return train_x, train_y, test_x, test_y


def make_data(dataset):
    train_x, train_y, test_x, test_y = [], [], [], []
    train_data = datasets.make(dataset)
    test_data = datasets.make(dataset, split='test')

    for _, data in enumerate(train_data):
        train_x.append(np.array(data[0]))
        train_y.append(data[1])

    for _, data in enumerate(test_data):
        test_x.append(np.array(data[0]))
        test_y.append(data[1])

    return 0


def save_dataset(path, dataset):
    train_x, train_y, test_x, test_y = dataset

    np.save(os.path.join(path, 'train_x.npy'), train_x)
    np.save(os.path.join(path, 'train_y.npy'), train_y)
    np.save(os.path.join(path, 'test_x.npy'), test_x)
    np.save(os.path.join(path, 'test_y.npy'), test_y)

    return 0
