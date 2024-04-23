from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The following code can be used to obtain a DataLoader for CIFAR10, ready for training in pytorch :

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader_full = DataLoader(c10train,batch_size=32,shuffle=True)
testloader_full = DataLoader(c10test,batch_size=32) 

### RandomSampler in order to use a subset of training :

## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)

### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.

### Training

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vgg
from utils import progress_bar

# Device configurationcd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = vgg.VGG('VGG11')

criterion = nn.CrossEntropyLoss()

# create your optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

model = model.to(device)

def train(trainloader, nb_epochs=2):
    accuracies=[]
    for epoch in range(nb_epochs): 
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        
        for i, (inputs, labels) in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                accuracy = 100.*correct/total
                accuracies.append(accuracy)

                progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(i+1), accuracy, correct, total))

        print('Finished Training')
    
    return max(accuracies)

### Accuracy vs Nb of Parameters 

import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the total number of parameters
nb_params = count_parameters(model)
print(f"Number of parameters in the model: {nb_params}")

def acc_nb_param(accuracy, nb_params):
    plt.plot(nb_params, accuracy, marker='o')
    plt.title('Accuracy vs Number of Parameters')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Top 1 Accuracy (%)')
    plt.grid(True)
    plt.show()

# Mode

TRAINING = False
SAVE_MODEL = False
LOAD_MODEL = True
INFERENCE = True

# Hyperparameters
nb_epochs = 100
current_train_loader = trainloader_full

# Train
if TRAINING:
    best_acc = train(current_train_loader, nb_epochs)
    print(f"Best accuracy is {best_acc:.3f}% over f{nb_epochs}")
    #best_acc = 87.500
    acc_nb_param(best_acc, nb_params)

### Save and Reload trained models

#model = MySuperModel(hyperparam = hparam_currentvalue)


if SAVE_MODEL :
    state = {
            'net': model.state_dict(),
            'hyperparam': 'VGG11'
    }
    print('Saving model')
    torch.save(state, 'mybestmodel.pth')

model_path = 'model/cirfar10_e100.pth'

if LOAD_MODEL :

    print('Loading model')

    # We load the dictionary
    loaded_cpt = torch.load(model_path)

    # Fetch the hyperparam value
    hparam_bestvalue = loaded_cpt['hyperparam']

    # Define the model 
    model = vgg.VGG(vgg_name = hparam_bestvalue)

    # Finally we can load the state_dict in order to load the trained parameters 
    model.load_state_dict(loaded_cpt['net'])

if INFERENCE : 

    print('Inference')

    # If you use this model for inference (= no further training), you need to set it into eval mode
    model.eval()

    # Move the model to the same device as the inputs
    model = model.to(device)

    # Iterate through the test data loader
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader_full:  # You can change to testloader_subset if needed
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
