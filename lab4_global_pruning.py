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

### Global Pruning, no retrain

# Pruning ratio
amount = 0.5

batch_size = 32
epochs = 100
model_path = os.path.join('model', model_name()+'.pth')
print('Model path', model_path)

# Create data loaders for training, validation, and test sets
trainloader, testloader = dataloader2(batch_size)

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Utilisation du GPU')


# Define the model, let's say it is called "mymodel"
architecture_name='ResNet18'
mymodel = ResNet18().to(device)
optimizer = optim.SGD(mymodel.parameters(), lr=0.01, momentum=0.9)
# Initialize the scheduler
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
        "model": model_path,
        "pruning ratio": amount
        }
    )

for epoch in range(epochs):
    print('\nEpoch: %d' % epoch)
    mymodel.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    ### During training (check the algorithm in the course and in the paper to see the exact sequence of operations)
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = mymodel(inputs)
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

    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = mymodel(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

        accuracy_val = 100. * correct_val / total_val
        val_accuracies.append(accuracy_val)

        print('Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
                % (val_loss / len(testloader), accuracy_val, correct_val, total_val))
        
        # Save the model if validation loss is minimized
        if  accuracy_val > best_val_acc:
            print('new best val accuracy:', accuracy_val)
            best_val_acc = accuracy_val
            torch.save(mymodel.state_dict(), model_path)
            print(f"\nModel with best accuracy saved as {model_path}")
        
        # Save validation loss for this epoch
        val_losses.append(val_loss / len(testloader))

    # Update the learning rate
    scheduler.step(val_loss / len(testloader))

    # Log metrics to wandb
    wandb.log({"Accuracy": accuracy_val, "Training loss": train_loss / len(trainloader), "Validation loss": val_loss / len(testloader), "Learning rate": optimizer.param_groups[0]['lr']}, step=epoch)

best_val_loss = min(val_losses)  # Find the best validation loss
best_val_loss_epoch = val_losses.index(best_val_loss)  # Find the epoch corresponding to the best validation loss

best_val_acc = max(val_accuracies)
best_val_acc_epoch = val_accuracies.index(best_val_acc)  

### Pruning after training
print('Pruning after training')

for idx, m in enumerate(mymodel.modules()): #  iterator over all modules in the network
    if hasattr(m, 'weight'):
        print(idx, '->', m)
        prune.l1_unstructured(m, name="weight", amount=amount)

print('Inference')
mymodel.eval()

# Move the model to the same device as the inputs
mymodel = model.to(device)

# Iterate through the test data loader
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:  # You can change to testloader_subset if needed
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = mymodel(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

# Calculate the accuracy
accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')
wandb.run.summary["Test Accuracy"] = accuracy

# Log the best validation loss and corresponding epoch
wandb.run.summary["best_validation_loss"] = best_val_loss
wandb.run.summary["best_validation_loss_epoch"] = best_val_loss_epoch
wandb.run.summary["best_accuracy"] = best_val_acc
wandb.run.summary["best_validation_acc_epoch"] = best_val_acc_epoch
           
wandb.finish()

