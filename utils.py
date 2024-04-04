import torch
from torch import nn
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
from PIL import Image

class CIFAR_Dataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
def get_data_loaders(dataset_path, batch_size, train_transform, test_transform):

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
    
    train_images = train_images.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)

    batch_dict = load_cifar_batch(os.path.join(dataset_path, f'test_batch'))
    test_images = np.array(batch_dict[b'data']) 
    test_labels = np.array(batch_dict[b'labels'])
    test_images = test_images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    
    train_dataset = CIFAR_Dataset(train_images,train_labels,transform=train_transform)
    test_dataset = CIFAR_Dataset(test_images,test_labels,transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def train(model, optimizer, loss_function, loader, DEVICE):

    loss = 0.0
    accuracy = 0.0
    count = 0

    model.train()

    for i, data in enumerate(loader):
        images, labels = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        predicted_output = model(images)
        fit = loss_function(predicted_output, labels)
        fit.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
        optimizer.step()
        loss += fit.item()
        _, predicted = torch.max(predicted_output, 1)
        accuracy += (predicted == labels).sum().item()
        count += len(predicted)
    
    loss = loss / len(loader)
    accuracy = accuracy / count

    return loss, accuracy


def test(model, loss_function, loader, DEVICE):

    loss = 0.0
    accuracy = 0.0
    count = 0

    model.eval()

    with torch.no_grad():
      for i, data in enumerate(loader):
          images, labels = data
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)
          predicted_output = model(images)
          fit = loss_function(predicted_output, labels)
          loss += fit.item()
          _, predicted = torch.max(predicted_output, 1)
          accuracy += (predicted == labels).sum().item()
          count += len(predicted)
    
    loss = loss / len(loader)
    accuracy = accuracy / count

    return loss, accuracy

def save_checkpoint(model, optimizer, epoch, filename='checkpoints/checkpoint.pth.tar'):
    state = {"model_state_dic": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}
    torch.save(state, filename)

def load_checkpoint(model, optimizer, file):
    checkpoint = torch.load(file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dic'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


