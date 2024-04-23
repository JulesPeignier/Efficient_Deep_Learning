import matplotlib.pyplot as plt
import datetime


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
