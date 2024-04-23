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

batch_size = 32

# Create data loaders for training, validation, and test sets
trainloader_full, valloader_full, testloader_full = dataloader(batch_size, train_val_split=0.8)

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Utilisation du GPU')


model_path = 'model/model_17-04-2024_15h28_best_acc.pth'

architecture_name='VGG11'

print('Loading model')

# Load dict
state_dict = torch.load(model_path)

# Define the model 
model = vgg.VGG(architecture_name)

# Finally we can load the state_dict in order to load the trained parameters 
model.load_state_dict(state_dict)

# quantization
model.half()  # convert all the model parameters to 16 bits half precision

print('Inference')

# If you use this model for inference (= no further training), you need to set it into eval mode
model.eval()

# Move the model to the same device as the inputs
model = model.to(device)

# Iterate through the test data loader
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader_full:  # You can change to testloader_subset if needed
        inputs, labels = inputs.half().to(device), labels.half().to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

# Calculate the accuracy
accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')
