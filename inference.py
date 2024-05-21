from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from data_prep import dataloader2
from resnet import ResNet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import progress_bar
from tools import *
import os

def model_inference(device, model, testloader, quantize=None):

    if isinstance(model, str):  # If model is a path to a state dictionary file
        # Load the model from the provided path
        print('Loading model')
        # Load dict
        state_dict = torch.load(model)

        # Define the model 
        model = ResNet18()

        # Finally we can load the state_dict in order to load the trained parameters 
        model.load_state_dict(state_dict)

    # If you use this model for inference (= no further training), you need to set it into eval mode
    model.eval()
    # Move the model to the same device as the inputs
    model = model.to(device)

    if quantize == 'Half':
        print('Half quantization')
        # quantization
        model.half()  # convert all the model parameters to 16 bits half precision
        print('Inference')
        # Iterate through the test data loader
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:  # You can change to testloader_subset if needed
                inputs, labels = inputs.half().to(device), labels.half().to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    else:
        print('No quantization')

        print('Inference')

        # Iterate through the test data loader
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:  # You can change to testloader_subset if needed
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%\n')

    return accuracy

def test():
    # Device configurationcd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Utilisation du GPU')

    batch_size = 32
    # Create data loaders for training, validation, and test sets
    trainloader, testloader = dataloader2(batch_size)

    model_path  = 'model/retrained/retrained_pruned_95percent_model_13-05-2024_13h41.pth'

    model_inference(device, model_path, testloader, quantize='Quarter')

# test()