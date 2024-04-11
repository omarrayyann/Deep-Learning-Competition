import torch
import numpy as np
from torch.autograd import Variable
import torch
import numpy as np
from utils import mixup_data

def train(model, optimizer, loss_function, loader, DEVICE):

    loss = 0.0
    accuracy = 0.0
    count = 0

    model.train()

    for _, data in enumerate(loader):
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


def mixup_train(model, optimizer, loss_function, loader, device):
    
    model.train()

    loss = 0.0
    accuracy = 0.0
    count = 0
    
    for _, data in enumerate(loader):
        
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        images, labels_a, labels_b, ratio = mixup_data(images, labels, device)
        images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))

        optimizer.zero_grad()
        predicted_output = model(images)

        fit = ratio * loss_function(predicted_output, labels_a) + (1 - ratio) * loss_function(predicted_output, labels_b)
        
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
      for _, data in enumerate(loader):
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
