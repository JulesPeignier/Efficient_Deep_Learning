import torch.optim as optim
from depthwise_separable_conv_resnet import *
from tools import *
import os
import wandb
from train import training

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Utilisation du GPU')

set_seed(444)  # Set your seed value here

# Training hyperparameters
wandb_log = True
epochs = 100
batch_size = 32
amounts = [0.4, 0.6, 0.7]
use_mixup = True

model_path = "model/distillation/dist_dsc_tinyresnet_in_14-05-2024_22h10.pth" # Accuracy: 91.41%
architecture_name = 'DSC_TinyResNet'

best_acc = 0
best_amount = None
results = []

for amount in amounts:

    # Load Trained Model
    print(f'Loading model: {model_path}')
    state_dict = torch.load(model_path)
    model = DSC_TinyResNet().to(device)
    model.load_state_dict(state_dict)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    retrained_model_path = os.path.join('model/retrained/', f'{round(amount * 100)}p_dist_dsc_tinyresnet_in_14-05-2024_22h10.pth')
    # retrained_model_path = os.path.join('model/retrained/', os.path.basename(model_path).replace('pruned_', 'retrained_pruned_'))
    print(f'Retrain model as {retrained_model_path} after {round(amount * 100)}% pruning')

    # Define optimizer, scheduler, and criterion again if needed
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    criterion = nn.CrossEntropyLoss()

    if wandb_log:
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
            "Mixup": use_mixup,
            "scheduler": scheduler,
            "model": retrained_model_path,
            "pruning ratio": amount
            }
        )

    # {
    #     "train_losses": train_losses,
    #     "val_losses": val_losses,
    #     "best_val_loss": best_val_loss,
    #     "best_val_loss_epoch": best_val_loss_epoch,
    #     "val_accuracies": val_accuracies,
    #     "best_val_acc": best_val_acc,
    #     "best_val_acc_epoch": best_val_acc_epoch,
    # }
    results_dict = training(
        model,
        device,
        epochs,
        batch_size,
        retrained_model_path,
        optimizer,
        scheduler,
        criterion,
        patience=100,
        wandb_log=wandb_log,
        use_mixup=use_mixup, # Added parameter for controlling mixup
        alpha=1.0,  
    )

    best_val_acc = results_dict['best_val_acc']
    if wandb_log:   
        wandb.finish()

    results.append((amount, best_val_acc))

    if best_val_acc > best_acc:
        best_acc = best_val_acc
        best_amount = amount

print(f'Best pruning amount {best_amount} with {best_acc}% accuracy after retrain')