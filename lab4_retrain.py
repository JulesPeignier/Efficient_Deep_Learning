import random
from inference import inference
from torchvision.datasets import CIFAR10
import numpy as np 
from torch.utils.data.dataloader import DataLoader
from data_prep import dataloader2
import torch.optim as optim
from resnet import ResNet18
from utils import progress_bar
from tools import *
import os
import wandb

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

### Retrain model after global pruning

# Pre-trained Model
model_path = 'model/model_07-05-2024_11h10.pth'
model_name = os.path.basename(model_path)
architecture_name = 'ResNet18'
retrained_model_path = os.path.join('model', 'retrained_pruned' + model_name)
print('retrained_model_path: ', retrained_model_path)
# Pruning ratio
amount = 0.5
# Training hyperparameters
batch_size = 32
epochs = 1

# Create data loaders for training, validation, and test sets
trainloader, testloader = dataloader2(batch_size)

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Utilisation du GPU')

print('Loading model to prune: ', model_name)
# Load dict
state_dict = torch.load(model_path)
# Define the model 
my_pretrained_model = ResNet18().to(device)
# Finally we can load the state_dict in order to load the trained parameters 
my_pretrained_model.load_state_dict(state_dict)
# Print the architecture of the pre-trained model
print(my_pretrained_model)

print('Pruning')
for idx, m in enumerate(my_pretrained_model.modules()): #  iterator over all modules in the network
    if hasattr(m, 'weight'):
        print(idx, '->', m)
        prune.l1_unstructured(m, name="weight", amount=amount)

# Retraining after pruning
print('Retraining after pruning')

# Define optimizer, scheduler, and criterion again if needed
optimizer = optim.SGD(my_pretrained_model.parameters(), lr=0.01, momentum=0.9)
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
        "model": retrained_model_path,
        "pruning ratio": amount
        }
    )

for epoch in range(epochs):
    print('\nEpoch: %d' % epoch)
    my_pretrained_model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    # Training loop
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = my_pretrained_model(inputs)
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
        my_pretrained_model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = my_pretrained_model(inputs)
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
torch.save(my_pretrained_model.state_dict(), retrained_model_path)
print(f"\nRetrained model saved as {retrained_model_path}")

# Log the best validation loss and corresponding epoch
wandb.run.summary["best_validation_loss"] = best_val_loss
wandb.run.summary["best_validation_loss_epoch"] = best_val_loss_epoch
wandb.run.summary["best_accuracy"] = best_val_acc
wandb.run.summary["best_validation_acc_epoch"] = best_val_acc_epoch
           
wandb.finish()