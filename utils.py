import torch
from torch import nn
from torch.utils.data import Dataset
import pickle
import numpy as np
import os

def combine_function(x, output, in_channels, out_channels, stride):
    layer = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
        nn.BatchNorm2d(out_channels)
    )
    x = layer(x)
    return x + output
    
class CIFAR_Dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
def get_data_loaders(dataset_path, train_percentage, batch_size):

    def load_cifar_batch(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    train_images = None
    train_labels = None

    for i in range(5):
        batch_dict = load_cifar_batch(os.path.join(dataset_path, f'data_batch_{i+1}'))
        train_images = np.array(batch_dict[b'data']) if train_images is None else np.concatenate((train_images, np.array(batch_dict[b'data'])))
        train_labels = np.array(batch_dict[b'labels']) if train_labels is None else np.concatenate((train_labels, np.array(batch_dict[b'labels'])))
    
    batch_dict = load_cifar_batch(os.path.join(dataset_path, f'test_batch'))
    train_images = np.array(batch_dict[b'data']) if train_images is None else np.concatenate((train_images, np.array(batch_dict[b'data'])))
    train_labels = np.array(batch_dict[b'labels']) if train_labels is None else np.concatenate((train_labels, np.array(batch_dict[b'labels'])))
    
    train_images = train_images.reshape((60000, 3, 32, 32)).transpose(0, 1, 2, 3)/255.0
    
    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()
    
    dataset = CIFAR_Dataset(train_images,train_labels)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*train_percentage), len(dataset)-int(len(dataset)*train_percentage)])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader