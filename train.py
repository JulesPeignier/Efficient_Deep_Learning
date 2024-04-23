from utils import progress_bar
import torch
import os
import wandb
import sys


def train(trainloader, valloader, nb_epochs, model, device, optimizer, scheduler, criterion, model_filename, batch_size):

    # Get the initial learning rate from the optimizer
    initial_lr = optimizer.param_groups[0]['lr']

    print('initial lr', initial_lr)
    print('sys.path', sys.path)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="VGG-perso",
        # track hyperparameters and run metadata
        config={
        "initial learning rate": initial_lr,  # Log the initial learning rate,
        "architecture": 'VGG11',
        "dataset": "CIFAR-10",
        "epochs": nb_epochs,
        "batch size": batch_size,
        "model": model_filename,
        }
    )

    val_accuracies = []
    best_val_acc = 0.0  # Track the best validation accuracy

    train_losses = []
    val_losses = []

    best_acc_model_filename = f"{model_filename}_best_acc.pth"
    best_loss_model_filename = f"{model_filename}_best_loss.pth"

    for epoch in range(nb_epochs): 
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
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
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

            accuracy_val = 100. * correct_val / total_val
            val_accuracies.append(accuracy_val)

            print('Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
                  % (val_loss / len(valloader), accuracy_val, correct_val, total_val))

            
            # Save the model if validation loss is minimized
            if epoch == 0 or (val_loss / len(valloader)) < min(val_losses):
                print('new best val loss:', val_loss / len(valloader))
                torch.save(model.state_dict(), os.path.join('model', best_loss_model_filename))
                print(f"\nModel with minimum validation loss saved as {best_loss_model_filename}")

            # Save the model if validation accuracy is improved
            elif accuracy_val > best_val_acc:
                print('new best val accuracy:', accuracy_val)
                best_val_acc = accuracy_val
                torch.save(model.state_dict(), os.path.join('model',best_acc_model_filename))
                print(f"\nModel with best accuracy saved as {best_acc_model_filename}")
            
            # Save validation loss for this epoch
            val_losses.append(val_loss / len(valloader))
        
        # Update the learning rate
        scheduler.step(val_loss / len(valloader))


        # Log metrics to wandb
        wandb.log({"Accuracy": accuracy_val, "Training loss": train_loss / len(trainloader), "Validation loss": val_loss / len(valloader), "Learning rate": optimizer.param_groups[0]['lr']}, step=epoch)

   

    best_val_loss = min(val_losses)  # Find the best validation loss
    best_val_loss_epoch = val_losses.index(best_val_loss)  # Find the epoch corresponding to the best validation loss

    best_val_acc = max(val_accuracies)
    best_val_acc_epoch = val_accuracies.index(best_val_acc)  

    # Log the best validation loss and corresponding epoch
    wandb.run.summary["best_validation_loss"] = best_val_loss
    wandb.run.summary["best_validation_loss_epoch"] = best_val_loss_epoch
    wandb.run.summary["best_accuracy"] = best_val_acc
    wandb.run.summary["best_validation_acc_epoch"] = best_val_acc_epoch

    

    return val_accuracies, train_losses, val_losses

