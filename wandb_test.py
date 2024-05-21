import wandb
import random
from tools import model_name
from train import train
from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
from data_prep import dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import ResNet18
from utils import progress_bar
from tools import *
from train import train
import os
from inference import inference

### Preprocessing

batch_sizes = [32]
# batch_sizes = [4, 8]

for batch_size in batch_sizes:

    # Create data loaders for training, validation, and test sets
    trainloader_full, valloader_full, testloader_full = dataloader(batch_size, train_val_split=0.8)

    # List of optimizers to iterate over
    # optimizers = [
    #     (optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    #     (optim.Adam, {"lr": 0.001}),
    #     (optim.RMSprop, {"lr": 0.001})
    # ]

    optimizers = [
        (optim.SGD, {"lr": 0.01, "momentum": 0.9})
    ]

    for optimizer_class, optimizer_params in optimizers:

        # Construct the filename with the timestamp
        model_filename = model_name()

        print(f'training {model_filename} with {optimizer_class.__name__ } {optimizer_params} and batch size {batch_size}')

        ### Training

        # Hyperparameters
        nb_epochs = 2

        # Device configurationcd
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('Utilisation du GPU')

        model = ResNet18()
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        model = model.to(device)

        # Initialize the scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        val_accuracies, train_losses, val_losses = train(trainloader_full, valloader_full, nb_epochs, model, device, optimizer, scheduler, criterion, model_filename, batch_size)


        model_filename = f"{model_filename}_best_loss.pth"
        model_path = os.path.join('model', model_filename)

        print('model path', model_path)

        inference(device, model_path, testloader_full, architecture_name='VGG11')

        wandb.finish()
