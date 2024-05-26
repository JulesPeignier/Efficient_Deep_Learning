import random
from inference import model_inference
from torchvision.datasets import CIFAR10
import numpy as np 
from torch.utils.data.dataloader import DataLoader
from data_prep import dataloader2
import torch.optim as optim
from resnet import ResNet18
from tiny_resnet import TinyResNet18
from depthwise_separable_conv_resnet import *
from utils import progress_bar
from tools import *
import os
import wandb
from pruning import global_pruning

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Utilisation du GPU')

# Create data loaders for training, validation, and test sets
trainloader, testloader = dataloader2(batch_size=32)

# Load Trained Model
model_path = 'model/distillation/retrained_pruned_95percent_dist_e300_14-05-2024_22h10.pth'
print(f'Loading model: {model_path}')
state_dict = torch.load(model_path)
model = DSC_MicroResNet() 
model.load_state_dict(state_dict)

batch_size = 32
amount = 0.75
pruned_model_path = global_pruning(model, model_path, amount)

# Load Pruned Model
state_dict = torch.load(pruned_model_path)
pruned_model = DSC_MicroResNet()
pruned_model.load_state_dict(state_dict)
pruned_model.to(device)

# Inference
print(f'{amount} percent Pruned Model Inference')
test_accuracy_pruned = model_inference(device, pruned_model, testloader)

### Retrain Model after Global Pruning

# Training hyperparameters
epochs = 300
architecture_name = 'DSC_MicroResNet'
retrained_model_path = os.path.join('model/retrained/', os.path.basename(pruned_model_path).replace('pruned_', 'retrained_pruned_'))
print(retrained_model_path)
print('Retraining after pruning')

# Define optimizer, scheduler, and criterion again if needed
optimizer = optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = nn.CrossEntropyLoss()

val_accuracies = []
best_val_acc = 0.0  # Track the best validation accuracy
train_losses = []
val_losses = []

wandb.init(
        # set the wandb project where this run will be logged
        project="VGG-perso",
        # track hyperparameters and run metadata
        config={
        "initial learning rate": 0.01,  # Log the initial learning rate,
        "architecture": architecture_name,
        "dataset": "CIFAR-10",
        "epochs": epochs,
        "batch size": batch_size,
        "scheduler": scheduler,
        "model": pruned_model_path,
        "pruning ratio": amount
        }
    )

for epoch in range(epochs):
    print('\nEpoch: %d' % epoch)
    pruned_model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    # Training loop
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = pruned_model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

        accuracy_train = 100. * correct_train / total_train

        progress_bar(i, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                        % (train_loss / (i + 1), accuracy_train, correct_train, total_train))

    # Save training loss for this epoch
    train_losses.append(train_loss / len(trainloader))

    # Validation loop
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        pruned_model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = pruned_model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    # Calculate accuracy
    accuracy_val = 100. * correct_val / total_val
    val_accuracies.append(accuracy_val)
    # Save validation loss for this epoch
    val_losses.append(val_loss / len(testloader))
    print('Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
          % (val_loss / len(testloader), accuracy_val, correct_val, total_val))

    # Update the learning rate
    scheduler.step(val_loss / len(testloader))

    # Log metrics to wandb
    wandb.log({"Accuracy": accuracy_val, "Training loss": train_loss / len(trainloader), "Validation loss": val_loss / len(testloader), "Learning rate": optimizer.param_groups[0]['lr']}, step=epoch)

best_val_loss = min(val_losses)  # Find the best validation loss
best_val_loss_epoch = val_losses.index(best_val_loss)  # Find the epoch corresponding to the best validation loss

best_val_acc = max(val_accuracies)
best_val_acc_epoch = val_accuracies.index(best_val_acc)

# Save the retrained model
torch.save(pruned_model.state_dict(), retrained_model_path)
print(f"\nRetrained model saved at {retrained_model_path}")

test_accuracy_retrained = model_inference(device, pruned_model, testloader)

# Log the best validation loss and corresponding epoch
wandb.run.summary["best_validation_loss"] = best_val_loss
wandb.run.summary["best_validation_loss_epoch"] = best_val_loss_epoch
wandb.run.summary["best_accuracy"] = best_val_acc
wandb.run.summary["best_validation_acc_epoch"] = best_val_acc_epoch

wandb.run.summary["test_accuracy_pruned"] = test_accuracy_pruned
wandb.run.summary["test_accuracy_retrained"] = test_accuracy_retrained
           
wandb.finish()