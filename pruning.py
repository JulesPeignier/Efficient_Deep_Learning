import torch
from resnet import ResNet18

import torch.nn as nn
import os
import torch
import torch.nn.utils.prune as prune

def global_pruning(model, model_path, amount):
    if isinstance(model, str):  # If model is a path to a state dictionary file
        # Load the model from the provided path
        print('Loading model before pruning')
        # Load dict
        state_dict = torch.load(model_path)
        # Define the model 
        model = ResNet18()
        # Finally we can load the state_dict in order to load the trained parameters 
        model.load_state_dict(state_dict)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    pruned_model_path = os.path.join(os.path.dirname(model_path), 
                                    'pruned/' + f'pruned_{round(amount * 100)}percent_' + os.path.basename(model_path))
    torch.save(model.state_dict(), pruned_model_path)
    
    print(f"Pruned Model saved at {pruned_model_path}\n")

    return pruned_model_path


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