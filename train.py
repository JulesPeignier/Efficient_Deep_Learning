import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import prune
from utils import progress_bar
from tiny_resnet import TinyResNet18
from resnet import ResNet18
from inference import model_inference
from tools import *
from data_prep import dataloader2
import wandb

class Train:
    def __init__(self, batch_size=32, epochs=1, model_name='ResNet18', amount=0.5, prune_after_training=False, inference_after_training=False, log_wandb=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.amount = amount
        self.prune_after_training = prune_after_training
        self.inference_after_training = inference_after_training
        self.log_wandb = log_wandb
        
        # Print model path
        self.model_path = os.path.join('model', model_name() +'.pth')
        print('Model path:', self.model_path)

        # Create data loaders for training, validation, and test sets
        self.trainloader, self.testloader = dataloader2(self.batch_size)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print('Utilisation du GPU')

        # Define the model
        if self.model_name == 'TinyResNet18':
            self.mymodel = TinyResNet18().to(self.device)
        elif self.model_name == 'ResNet18':
            self.mymodel = ResNet18().to(self.device)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        # Initialize optimizer, scheduler, and criterion
        self.optimizer = optim.SGD(self.mymodel.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize WandB
        if self.log_wandb:
            wandb.init(project="VGG-perso",
                       config={"initial learning rate": 0.01, "architecture": self.model_name,
                               "dataset": "CIFAR-10", "epochs": self.epochs, "batch size": self.batch_size,
                               "model": self.model_path, "pruning ratio": self.amount})

    def train(self):
        val_accuracies = []
        best_val_acc = 0.0  # Track the best validation accuracy
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            print('\nEpoch:', epoch)
            self.mymodel.train()
            train_loss = 0
            correct_train = 0
            total_train = 0

            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.mymodel(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                accuracy_train = 100. * correct_train / total_train

                progress_bar(i, len(self.trainloader),
                             'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss / (i + 1), accuracy_train,
                                                                                correct_train, total_train))

            train_losses.append(train_loss / len(self.trainloader))

            val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                self.mymodel.eval()
                for inputs, labels in self.testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.mymodel(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

                accuracy_val = 100. * correct_val / total_val
                val_accuracies.append(accuracy_val)

                print('Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)' % (val_loss / len(self.testloader), accuracy_val,
                                                                    correct_val, total_val))

                if accuracy_val > best_val_acc:
                    print('New best val accuracy:', accuracy_val)
                    best_val_acc = accuracy_val
                    torch.save(self.mymodel.state_dict(), self.model_path)
                    print(f"Model with best accuracy saved at {self.model_path}")

                val_losses.append(val_loss / len(self.testloader))

            self.scheduler.step(val_loss / len(self.testloader))

            # Log metrics to wandb
            if self.log_wandb:
                wandb.log({"Accuracy": accuracy_val, "Training loss": train_loss / len(self.trainloader),
                           "Validation loss": val_loss / len(self.testloader),
                           "Learning rate": self.optimizer.param_groups[0]['lr']}, step=epoch)

        best_val_loss = min(val_losses)  # Find the best validation loss
        best_val_loss_epoch = val_losses.index(best_val_loss)  # Find the epoch corresponding to the best validation loss
        best_val_acc = max(val_accuracies)
        best_val_acc_epoch = val_accuracies.index(best_val_acc)

        # Log the best validation loss and corresponding epoch
        if self.log_wandb:
            wandb.run.summary["best_validation_loss"] = best_val_loss
            wandb.run.summary["best_validation_loss_epoch"] = best_val_loss_epoch
            wandb.run.summary["best_accuracy"] = best_val_acc
            wandb.run.summary["best_validation_acc_epoch"] = best_val_acc_epoch

        # Pruning after training
        if self.prune_after_training:
            print('Pruning after training')
            for idx, m in enumerate(self.mymodel.modules()):
                if hasattr(m, 'weight'):
                    # print(idx, '->', m)
                    prune.l1_unstructured(m, name="weight", amount=self.amount)

        # Inference after training or pruning
        if self.inference_after_training or self.prune_after_training:
            inference_accuracy = model_inference(self.device, self.mymodel, self.testloader, quantize=None)
            if self.log_wandb:
                wandb.run.summary["Test Accuracy"] = inference_accuracy

        if self.log_wandb:
            wandb.finish()

