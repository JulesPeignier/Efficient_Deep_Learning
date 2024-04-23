from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from data_prep import dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vgg
from utils import progress_bar
from tools import *
from train import train
import os
import wandb

def inference(device, model_path, testloader, architecture_name='VGG11'):
    
    print('Loading model')

    # # We load the dictionary
    # loaded_cpt = torch.load(model_path)

    # # Fetch the hyperparam value
    # hparam_bestvalue = loaded_cpt['hyperparam']

    # Load dict
    state_dict = torch.load(model_path)

    # Define the model 
    model = vgg.VGG(architecture_name)

    # Finally we can load the state_dict in order to load the trained parameters 
    model.load_state_dict(state_dict)

    print('Inference')

    # If you use this model for inference (= no further training), you need to set it into eval mode
    model.eval()

    # Move the model to the same device as the inputs
    model = model.to(device)

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
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    wandb.run.summary["Test Accuracy"] = accuracy

