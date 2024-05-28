import matplotlib.pyplot as plt
import datetime
import random
import numpy as np 
import torch
import wandb
import os
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def acc_nb_param(accuracy, nb_params):
    plt.plot(nb_params, accuracy, marker='o')
    plt.title('Accuracy vs Number of Parameters')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Top 1 Accuracy (%)')
    plt.grid(True)
    plt.show()

def model_name():
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time as a string
    timestamp = now.strftime("%d-%m-%Y_%Hh%M")

    # Construct the filename with the timestamp
    model_filename = f"model_{timestamp}"

    return model_filename

def plot_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_wandb(config):
    try:
        wandb.init(
            project="VGG-perso",
            config=config,
        )
    except BrokenPipeError:
        print("W&B initialization failed due to BrokenPipeError.")

def log_to_wandb(metrics, epoch):
    try:
        wandb.log(metrics, step=epoch)
    except BrokenPipeError:
        print("W&B logging failed due to BrokenPipeError.")

def save_json(data, path):
    """Save a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {path}")