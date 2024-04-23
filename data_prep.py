from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

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

def dataloader(batch_size=32, train_val_split=0.8):

    # Define the sizes for training and validation sets
    train_size = int(train_val_split * len(c10train))
    val_size = len(c10train) - train_size

    print('train_size', train_size)
    print('val_size', val_size)

    # Split the dataset
    train_set, val_set = random_split(c10train, [train_size, val_size])

    # Create data loaders for training, validation, and test sets
    trainloader_full = DataLoader(train_set, batch_size, shuffle=True)
    valloader_full = DataLoader(val_set, batch_size)
    testloader_full = DataLoader(c10test, batch_size)

    return trainloader_full, valloader_full, testloader_full

def dataloader2(batch_size=32):

    # Create data loaders for training, validation, and test sets
    trainloader = DataLoader(c10train, batch_size, shuffle=True)
    testloader = DataLoader(c10test, batch_size)

    return trainloader, testloader