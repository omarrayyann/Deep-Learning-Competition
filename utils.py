import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os

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
        return sample.float(), label
    
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
    
    train_images = train_images.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)/255

    batch_dict = load_cifar_batch(os.path.join(dataset_path, f'test_batch'))
    test_images = np.array(batch_dict[b'data']) 
    test_labels = np.array(batch_dict[b'labels'])
    test_images = test_images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)/255
    
    train_dataset = CIFAR_Dataset(train_images,train_labels,transform=train_transform)
    test_dataset = CIFAR_Dataset(test_images,test_labels,transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader

def save_checkpoint(model, optimizer, epoch, filename='checkpoints/checkpoint.pth.tar'):
    state = {"model_state_dic": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}
    torch.save(state, filename)

def load_checkpoint(model, optimizer, file):
    checkpoint = torch.load(file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dic'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def mixup_data(images, labels, device):
    
    index = torch.randperm(images.shape[0]).to(device)
    mix_ratio = np.random.beta(1, 1)

    mixed_images = mix_ratio * images + (1 - mix_ratio) * images[index, :]
    labels_a, labels_b = labels, labels[index]

    mixed_images.to(device)
    labels_a.to(device)
    labels_b.to(device)
    
    return mixed_images, labels_a, labels_b, mix_ratio