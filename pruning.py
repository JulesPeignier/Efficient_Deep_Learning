import torch
from resnet import ResNet18
from depthwise_separable_conv_resnet import *
from inference import *
from tools import *

import torch.nn as nn
import os
import torch
import torch.nn.utils.prune as prune

def global_pruning(model, pruned_model_path, amount):
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    torch.save(model.state_dict(), pruned_model_path)
    
    print(f"Pruned Model saved at {pruned_model_path}\n")



def compute_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            sparsity = 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
            print("Sparsity in {}: {:.2f}%".format(name, sparsity))
    # Calculate global sparsity
    total_non_zero = sum(torch.sum(module.weight != 0).item() for name, module in model.named_modules()
                        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear))
    total_elements = sum(module.weight.nelement() for name, module in model.named_modules()
                        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear))
    global_sparsity = 100. * (1 - total_non_zero / total_elements)
    print("Global sparsity: {:.2f}%".format(global_sparsity))
        
    return global_sparsity

def test():
    # Example usage:
    model_path = 'model/model_13-05-2024_13h41.pth'
    print(f'Loading model: {model_path}')
    # Load dict
    state_dict = torch.load(model_path)
    # Define the model 
    model = ResNet18()
    # Finally we can load the state_dict in order to load the trained parameters 
    model.load_state_dict(state_dict)

    sparsity = compute_sparsity(model)
    print("Sparsity og model:", sparsity, '\n')

    # Global Pruning
    amount = 0.625
    pruned_model_path = global_pruning(model, model_path, amount)

    # Load dict
    state_dict = torch.load(pruned_model_path)
    # Define the model 
    pruned_model = ResNet18()
    # Finally we can load the state_dict in order to load the trained parameters 
    pruned_model.load_state_dict(state_dict)

    sparsity = compute_sparsity(pruned_model)
    print("Sparsity pruned model:", sparsity, '\n')


def try_pruning_ratio():

    set_seed(444)

    amounts = [0.0, 0.1, 0.2, 0.22, 0.24, 0.25, 0.26, 0.28, 0.3]
    results = []

    # Device configurationcd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Utilisation du GPU')

    batch_size = 32
    # Create data loaders for training, validation, and test sets
    trainloader, testloader = dataloader2(batch_size)

    # model_path  = "model/distillation/dist_dsc_tinyresnet_in_14-05-2024_22h10.pth" #Accuracy on the test set: 91.41%
    model_path = 'model/retrained/60p_dist_dsc_tinyresnet_in_14-05-2024_22h10.pth' # 90.90%
    # Load dict
    state_dict = torch.load(model_path) 

    for amount in amounts:

        print('Pruning with amount', amount)
    
        model = DSC_TinyResNet().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Pruning
        if amount > 0:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    prune.remove(module, 'weight')

        # Iterate through the test data loader
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:  # You can change to testloader_subset if needed
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate the accuracy
        accuracy = 100 * correct / total
        print(f'Accuracy on the test set: {accuracy:.2f}%\n')

        results.append((amount, accuracy))
    
    return results

# print(try_pruning_ratio())


def prune_save():
    set_seed(444)
    # Device configurationcd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Utilisation du GPU')

    model_path = 'model/retrained/60p_dist_dsc_tinyresnet_in_14-05-2024_22h10.pth'
    # Load dict
    state_dict = torch.load(model_path) 
    model = DSC_TinyResNet().to(device)
    model.load_state_dict(state_dict)
    pruned_model_path = 'model/final/pruned_dist_dsc_tinyresnet_14-05-2024_22h10.pth'
    amount = 0.25
    global_pruning(model, pruned_model_path, amount)