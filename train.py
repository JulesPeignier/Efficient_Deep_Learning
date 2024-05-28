import random
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils.data.dataloader import DataLoader
from data_prep import dataloader2, mixup_criterion, mixup_data
import torch.optim as optim
from resnet import ResNet18
from tiny_resnet import TinyResNet18, MicroResNet
from depthwise_separable_conv_resnet import *
from utils import progress_bar
from tools import *
import os
import wandb
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F


# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision.transforms as transforms

# # def imshow(images, titles):
# #     num_images = len(images)
# #     fig, axes = plt.subplots(1, num_images, figsize=(15, 5))  # Adjust figsize as needed
# #     for i in range(num_images):
# #         img = images[i].cpu().numpy()
# #         img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
# #         img = np.clip(img, 0, 1)  # Clip values to [0, 1] for imshow
# #         axes[i].imshow(img)
# #         axes[i].set_title(titles[i])
# #         axes[i].axis('off')
# #     plt.savefig(os.path.join('data/Images/', 'train_mixup.png'))  # Save the image to a file
# #     plt.close()  # Close the plot to avoid displaying it


def training(
    model,
    device,
    epochs,
    batch_size,
    model_path,
    optimizer,
    scheduler,
    criterion,
    patience=10,
    wandb_log=True,
    use_mixup=False,  # Added parameter for controlling mixup
    alpha=1.0,
):

    trainloader, testloader = dataloader2(batch_size)

    val_accuracies = []
    best_val_acc = 0.0  # Track the best validation accuracy
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")  # Early stopping
    epochs_no_improve = 0

    for epoch in range(epochs):
        print("\nEpoch: %d" % epoch)
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            if use_mixup:  # if epoch > 30 and use_mixup:
                # Apply mixup
                inputs, labels_a, labels_b, lam, x, x_perm = mixup_data(
                    inputs, labels, alpha=alpha
                )
                ## Visualize mixup results for the first batch
                # if i == 0:
                #     img_grid = inputs[0]
                #     images = [img_grid, x[0], x_perm[0]]
                #     title = f'Mixup - Lambda: {lam:.2f}'
                #     titles = [title, 'img1', 'img2']
                #     imshow(images, titles)
                #     print(f'Label A: {labels_a[0].item()}, Label B: {labels_b[0].item()}')

                optimizer.zero_grad()
                outputs = model(inputs)
                # print('output', outputs[0])
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                # print('loss', loss)
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
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

        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
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
                torch.save(model.state_dict(), model_path)
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
            log_to_wandb(
                {
                    "Accuracy": accuracy_val,
                    "Training loss": train_loss / len(trainloader),
                    "Validation loss": val_loss / len(testloader),
                    "Learning rate": optimizer.param_groups[0]["lr"],
                },
                epoch,
            )

    best_val_loss = min(val_losses)
    best_val_loss_epoch = val_losses.index(best_val_loss)
    best_val_acc = max(val_accuracies)
    best_val_acc_epoch = val_accuracies.index(best_val_acc)

    if wandb_log:
        try:
            wandb.run.summary["best_validation_loss"] = best_val_loss
            wandb.run.summary["best_validation_loss_epoch"] = best_val_loss_epoch
            wandb.run.summary["best_accuracy"] = best_val_acc
            wandb.run.summary["best_validation_acc_epoch"] = best_val_acc_epoch
        except BrokenPipeError:
            print("W&B logging failed due to BrokenPipeError.")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "val_accuracies": val_accuracies,
        "best_val_acc": best_val_acc,
        "best_val_acc_epoch": best_val_acc_epoch,
    }


def main():

    set_seed(444)  # Set your seed value here

    wandb_log = True

    batch_size = 32
    epochs = 300
    use_mixup = True
    alpha = 0.2

    model_path = os.path.join("model", model_name() + ".pth")
    print("Model path", model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Utilisation du GPU")

    # Define the model
    architecture_name = "DSC_TinyResNet"  # architecture_name='ResNet18'
    mymodel = DSC_TinyResNet().to(device)  # mymodel = ResNet18().to(device)

    optimizer = optim.SGD(mymodel.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=1563, epochs=epochs)
    criterion = nn.CrossEntropyLoss()

    if wandb_log:
        config = {
            "initial learning rate": 0.01,
            "architecture": architecture_name,
            "dataset": "CIFAR-10",
            "epochs": epochs,
            "batch size": batch_size,
            "Mixup": use_mixup,
            "scheduler": scheduler,
            "model": model_path,
        }
        initialize_wandb(config)

    training_results = training(
        mymodel,
        device,
        epochs,
        batch_size,
        model_path,
        optimizer,
        scheduler,
        criterion,
        patience=20,
        wandb_log=wandb_log,
        use_mixup=use_mixup,
        alpha=alpha,
    )

    path_json = os.path.join("data/json", os.path.basename(model_path).strip('.pth') + '.json')
    print(f'Training results ave at: {path_json}')
    save_json(training_results, path_json)

    if wandb_log:
        try:
            wandb.finish()
        except BrokenPipeError:
            print("W&B logging failed due to BrokenPipeError.")


if __name__ == "__main__":
    main()
