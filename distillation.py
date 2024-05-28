import torch
import torch.nn.functional as F
from torch import nn, optim
from resnet import ResNet18
from tiny_resnet import *
from depthwise_separable_conv_resnet import *
from data_prep import dataloader2, mixup_criterion, mixup_data
from utils import progress_bar
from tools import *
import os
import wandb

def training(
    teacher_model,
    student_model,
    device,
    epochs,
    batch_size,
    model_path,
    optimizer,
    scheduler,
    criterion,
    distillation_alpha=0.5,
    T=2.0,
    patience=10,
    wandb_log=True,
):

    trainloader, testloader = dataloader2(batch_size)

    val_accuracies = []
    best_val_acc = 0.0  # Track the best validation accuracy
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # Early stopping
    epochs_no_improve = 0

    for epoch in range(epochs):
        print("\nEpoch: %d" % epoch)
        teacher_model.eval()
        student_model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Teacher and student outputs
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)

            # Compute the distillation loss
            loss_ce = criterion(student_outputs, labels)
            loss_kl = F.kl_div(
                F.log_softmax(student_outputs / T, dim=1),
                F.softmax(teacher_outputs / T, dim=1),
                reduction='batchmean'
            ) * (T * T)
            loss = distillation_alpha * loss_ce + (1 - distillation_alpha) * loss_kl

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            accuracy_train = 100.0 * correct_train / total_train

            progress_bar(
                i,
                len(trainloader),
                "Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)"
                % (train_loss / (i + 1), accuracy_train, correct_train, total_train),
            )

        train_losses.append(train_loss / len(trainloader))

        student_model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = student_model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

            accuracy_val = 100.0 * correct_val / total_val
            val_accuracies.append(accuracy_val)

            print(
                "Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)"
                % (val_loss / len(testloader), accuracy_val, correct_val, total_val)
            )

            # Save the model if new best accuracy
            if accuracy_val > best_val_acc:
                print("new best val accuracy:", accuracy_val)
                best_val_acc = accuracy_val
                torch.save(student_model.state_dict(), model_path)
                print(f"\nModel with best accuracy saved as {model_path}")

            val_losses.append(val_loss / len(testloader))

            # Check for early stopping
            if val_loss / len(testloader) < best_val_loss:
                best_val_loss = val_loss / len(testloader)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Update the learning rate
        scheduler.step(val_loss / len(testloader))

        if wandb_log:
            # Log metrics to wandb
            wandb.log(
                {
                    "Accuracy": accuracy_val,
                    "Training loss": train_loss / len(trainloader),
                    "Validation loss": val_loss / len(testloader),
                    "Learning rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

    best_val_loss = min(val_losses)
    best_val_loss_epoch = val_losses.index(best_val_loss)
    best_val_acc = max(val_accuracies)
    best_val_acc_epoch = val_accuracies.index(best_val_acc)

    if wandb_log:
        wandb.run.summary["best_validation_loss"] = best_val_loss
        wandb.run.summary["best_validation_loss_epoch"] = best_val_loss_epoch
        wandb.run.summary["best_accuracy"] = best_val_acc
        wandb.run.summary["best_validation_acc_epoch"] = best_val_acc_epoch

    return (
        train_losses,
        val_losses,
        best_val_loss,
        best_val_loss_epoch,
        val_accuracies,
        best_val_acc,
        best_val_acc_epoch,
    )

def main():

    wandb_log = True

    # Distillation parameters
    distillation_alpha=0.7
    T=20

    # Training hyperparameters
    batch_size = 32
    epochs = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Utilisation du GPU")

    ### Define the models
    # Load Teacher model
    teacher_architecture_name = 'TinyResNet18'
    teacher_model_path = "model/retrained/retrained_pruned_95percent_model_14-05-2024_22h10.pth" # Accuracy on the test set: 92.40%
    print(f'Loading teacher model: {teacher_model_path}')
    teacher_model = TinyResNet18().to(device)  
    teacher_model.load_state_dict(torch.load(teacher_model_path)) 

    # Create Student model
    student_architecture_name = 'DSC_MicroResNet'
    student_model_path = os.path.join('model/distillation/', os.path.basename(teacher_model_path).replace('model_', 'dist_e300_'))
    print(f'Student model: {student_architecture_name}')
    student_model = DSC_MicroResNet().to(device)  

    optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=20)
    criterion = nn.CrossEntropyLoss()

    if wandb_log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="VGG-perso",
            # track hyperparameters and run metadata
            config={
                "initial learning rate": 0.01,
                "Student architecture": student_architecture_name,
                "Student model": student_model_path,
                "Teacher architecture": teacher_architecture_name, 
                "Teacher model": teacher_model_path,
                "dataset": "CIFAR-10",
                "epochs": epochs,
                "batch size": batch_size,
                "scheduler": scheduler,
                "distillation_alpha": distillation_alpha,
                "T": T,
            },
        )
    training(
        teacher_model,
        student_model,
        device,
        epochs,
        batch_size,
        student_model_path,
        optimizer,
        scheduler,
        criterion,
        distillation_alpha,
        T,
        patience=300,
        wandb_log=wandb_log,
    )
    if wandb_log: 
        wandb.finish()

if __name__ == "__main__":
    main()

def tune_distillation_params():

    wandb_log = True

    # Distillation parameters
    distillation_alphas =  [0.5, 0.7, 0.9]
    temperatures = [5, 10, 20]

    # Training hyperparameters
    batch_size = 32
    epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Utilisation du GPU")

    ### Define the models
    # Load Teacher model
    teacher_architecture_name = 'TinyResNet18'
    teacher_model_path = "model/retrained/retrained_pruned_95percent_model_14-05-2024_22h10.pth" # Accuracy on the test set: 92.40%
    print(f'Loading teacher model: {teacher_model_path}')
    teacher_model = TinyResNet18().to(device)  
    teacher_model.load_state_dict(torch.load(teacher_model_path)) 

    # Create Student model
    student_architecture_name = 'DSC_MicroResNet'
    student_model_path = os.path.join('model/distillation/', os.path.basename(teacher_model_path).replace('model_', 'dist_'))
    print(f'Student model: {student_architecture_name}')

    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_params = (0, 0)
    results = []
    for alpha in distillation_alphas:
        for temp in temperatures:

            # Reinitialize the student model
            student_model = DSC_NanoResNet().to(device)

            print(f"Training with distillation_alpha={alpha}, temperature={temp}")
            optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=8)
            if wandb_log:
                wandb.init(
                project="VGG-perso",
                config={    
                "initial learning rate": 0.01,
                "Student architecture": student_architecture_name,
                "Student model": student_model_path,
                "Teacher architecture": teacher_architecture_name, 
                "Teacher model": teacher_model_path,
                "dataset": "CIFAR-10",
                "epochs": epochs,
                "batch size": batch_size,
                "scheduler": scheduler,
                "distillation_alpha": alpha,
                "T": temp,
                },
            )
            _, _, _, _, val_accuracies, best_val_acc, _ = training(
                teacher_model,
                student_model,
                device,
                epochs,
                batch_size,
                student_model_path,
                optimizer,
                scheduler,
                criterion,
                distillation_alpha=alpha,
                T=temp,
                patience=20,
                wandb_log=wandb_log,
            )
            if wandb_log:
                wandb.finish()

            results.append((alpha, temp, best_val_acc))

            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_params = (alpha, temp)

    print(f"Best accuracy: {best_acc} with distillation_alpha={best_params[0]}, temperature={best_params[1]}")
    print(f"\n Results: {results}")
