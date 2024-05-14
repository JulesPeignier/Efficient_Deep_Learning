import torch
from data_prep import dataloader2
from inference import path_inference, model_inference
from resnet import ResNet18
import torch.nn.utils.prune as prune

import torch.nn as nn

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Utilisation du GPU')

# Create data loaders for training, validation, and test sets
trainloader, testloader = dataloader2(batch_size=32)

# Pruning ratio
amounts = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.750]

model_path = 'model/model_13-05-2024_13h41.pth'

print('Inference after training')
path_inference(device, model_path, testloader)

accuracies = []

for amount in amounts :
    print('Loading model')
    # Load dict
    state_dict = torch.load(model_path)
    # Define the model 
    mymodel = ResNet18()
    # Finally we can load the state_dict in order to load the trained parameters 
    mymodel.load_state_dict(state_dict)
 
    for name, module in mymodel.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

    print(f'Inference after {amount*100}% pruning')
    accuracies.append(model_inference(device, mymodel, testloader))

print('ratio', amounts)
print('accuracies', accuracies)