from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import os

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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

def dataloader2(batch_size=32):

    # Create data loaders for training, validation, and test sets
    trainloader = DataLoader(c10train, batch_size, shuffle=True)
    testloader = DataLoader(c10test, batch_size)

    return trainloader, testloader

# rootdir = './data/cifar10'

# c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)

# trainloader = DataLoader(c10train,batch_size=9,shuffle=False) ### Shuffle to False so that we always see the same images
# trainloader = DataLoader(c10train,batch_size=9,shuffle=False) ### Shuffle to False so that we always see the same images

# from matplotlib import pyplot as plt 

# ### Let's do a figure for each batch
# f = plt.figure(figsize=(10,10))

# for i,(data,target) in enumerate(trainloader):
#     print(target)
#     data = (data.numpy())
#     print(data.shape)
#     plt.subplot(3,3,1)
#     plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(3,3,2)
#     plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(3,3,3)
#     plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(3,3,4)
#     plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))
    
#     plt.subplot(3,3,5)
#     plt.imshow(data[4].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(3,3,6)
#     plt.imshow(data[5].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(3,3,7)
#     plt.imshow(data[6].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(3,3,8)
#     plt.imshow(data[7].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(3,3,9)
#     plt.imshow(data[8].swapaxes(0,2).swapaxes(0,1))

#     break
# plt.show()
# # Save the figure
# plt.savefig(os.path.join('data/Images/', 'train_DA.png'))
# plt.close()