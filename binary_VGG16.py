# # References:
# Torchvision models: https://pytorch.org/vision/stable/models.html
# VGG16 pre-trained: https://worksheets.codalab.org/worksheets/0xe2ac460eee7443438d5ab9f43824a819
# How to freeze the layers:
# https://androidkt.com/pytorch-freeze-layer-fixed-feature-extractor-transfer-learning/
# https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
# https://debuggercafe.com/transfer-learning-with-pytorch/

# Danni Chen\09/24/2021


import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt
import sys

#torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
# Import torch.nn which contains all functions necessary to define a convolutional neural network
import torch.nn as nn
# Import NiFTIDataset to access the train_test_split method and the NiFTIDataset class
from utils.loading.NiFTIDataset import train_test_split, NiFTIDataset
from utils.loading.NiFTIDataset import NiFTIDataset
from utils.transforms.torchvision import Repeat, Rescale, Unsqueeze


## Retrieve Dataset from Metadata Dataframe and Load with Dataloader
# MetaData dataframe
metadata = pd.read_csv("Metadata/metadata.csv")

# Construct the appropriate transforms needed in the neural net.
# Normalization follows guidelines in https://pytorch.org/vision/stable/models.html.
# Rescale the image to (0,1), then convert to 3-channel grayscale, then normalize
# It in accordance with how it should be done using the above link.
transform = transforms.Compose([
    Rescale(0,1),
    Unsqueeze(0),
    Repeat(3,1,1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

# Retrieve the dataset from info obtained in metadata dataframe
dataset = NiFTIDataset(metadata=metadata,root='.',transform=transform)

# Split a NiFTIDatset into two groups (training and testing) based on information specified within its metadata dataframe
# Return a tuple containing two NiFTIDataset objects with training and testing data, respectively.
(training_data,testing_data) = train_test_split(dataset)

print('Number of data in the training dataset: ' + str(len(training_data)))
print('Number of data in the testing dataset: ' + str(len(testing_data)) + '\n')

# Visualize the 886th image in the training dataset
t = training_data.__getitem__(886)
print(t)

# load the data with dataloader
train_dataloader = DataLoader(training_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(testing_data,batch_size=32,shuffle=False)

plt.imshow(Rescale(0,1)(t['image'].T))

## Load the Data Into the Model
from utils.binary_VGG16_transfer_learning import binary_VGG16_transfer_learning

# Initialize a pre-trained VGG16 object will 
# download its weights to a cache directory.
model = models.vgg16(pretrained=True)

VGG16 = binary_VGG16_transfer_learning(model = model, 
                                train_dataloader = train_dataloader, 
                                test_dataloader = test_dataloader,
                                criterion = nn.CrossEntropyLoss(), 
                                optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9) )

running_loss = VGG16.model_training(numOfEpoch = 100)
print(running_loss)

accuracy = VGG16.model_testing()
print(accuracy)
