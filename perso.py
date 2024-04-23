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


### Preprocessing

# Create data loaders for training, validation, and test sets

trainloader_full, valloader_full, testloader_full = dataloader(train_val_split=0.8)

### Training

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Utilisation du GPU')
model = vgg.VGG('VGG11')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
model = model.to(device)

# Mode
TRAINING = False
PLOT = False
#SAVE_MODEL = True
LOAD_MODEL = True
INFERENCE = True

# Print the total number of parameters
nb_params = count_parameters(model)
print(f"\nNumber of parameters in the model: {nb_params}")

# Construct the filename with the timestamp
model_filename = model_name()

# Hyperparameters
nb_epochs = 2
current_train_loader = trainloader_full
current_val_loader = valloader_full

# Train
if TRAINING:
    train_accuracies, val_accuracies, train_losses, val_losses = train(current_train_loader, current_val_loader, nb_epochs, model, device, optimizer, criterion, model_filename)
    best_acc = max(val_accuracies)
    print(f"\n----> Best validation accuracy is {best_acc:.3f}% over {nb_epochs} epochs")
    #best_acc = 87.500
    if PLOT:
        acc_nb_param(best_acc, nb_params)
        plot_loss(train_losses, val_losses)

# if SAVE_MODEL :
#     state = {
#             'net': model.state_dict(),
#             'hyperparam': 'VGG11'
#     }
#     print('Saving model')
#     torch.save(state, 'mybestmodel.pth')

model_path = 'model/model_17-04-2024_18h31_best_acc.pth'

if LOAD_MODEL :

    print('Loading model')

    # # We load the dictionary
    # loaded_cpt = torch.load(model_path)

    # # Fetch the hyperparam value
    # hparam_bestvalue = loaded_cpt['hyperparam']

    # Load dict
    state_dict = torch.load(model_path)

    # Define the model 
    model = vgg.VGG(vgg_name = 'VGG11')

    # Finally we can load the state_dict in order to load the trained parameters 
    model.load_state_dict(state_dict)

if INFERENCE : 

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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
