# # References:
# Torchvision models: https://pytorch.org/vision/stable/models.html
# VGG16 pre-trained: https://worksheets.codalab.org/worksheets/0xe2ac460eee7443438d5ab9f43824a819
# How to freeze the layers:
# https://androidkt.com/pytorch-freeze-layer-fixed-feature-extractor-transfer-learning/
# https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
# https://debuggercafe.com/transfer-learning-with-pytorch/

# Danni Chen\09/24/2021

import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt
import socket
import time
import os

#torch
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
dataset = NiFTIDataset(metadata=metadata,root='/datadrive/COVID_IEEE/',transform=transform,
    slice_cols='Slice_25_Path')

# Split a NiFTIDatset into two groups (training and testing) based on information specified within its metadata dataframe
# Return a tuple containing two NiFTIDataset objects with training and testing data, respectively.
(training_data,testing_data) = train_test_split(dataset)

print('Number of data in the training dataset: ' + str(len(training_data)))
print('Number of data in the testing dataset: ' + str(len(testing_data)) + '\n')

# Visualize the 886th image in the training dataset
# t = training_data.__getitem__(886)
# print(t)

# load the data with dataloader
train_dataloader = DataLoader(training_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(testing_data,batch_size=32,shuffle=False)

# plt.imshow(Rescale(0,1)(t['image'].T))

## Load the Data Into the Model
from utils.binary_VGG16_transfer_learning import binary_VGG16_transfer_learning

# Initialize a pre-trained VGG16 object will 
# download its weights to a cache directory.
model = models.vgg16(pretrained=True)

# Select a device.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Make a directory to save information in.
log_dir = time.strftime(
    './runs/%b%d%y_%H-%M-%S_{}_binary/'.format(socket.gethostname()),
    time.localtime()
)

# Prepare the model.
# Freeze training for all layers
# To save computation time and that the network would already 
# be able to extract generic features from the dataset.
for param in model.features.parameters():
    param.requires_grad = False  

# https://androidkt.com/pytorch-freeze-layer-fixed-feature-extractor-transfer-learning/
# Remove the original fully-connected layer (the last layer) and create a new one
# Newly created modules have requires_grad=True by default
num_features = model.classifier[-1].in_features
classifier_layers = list(model.classifier.children())[:-1] # Remove the last layer
classifier_layers.extend([nn.Linear(in_features = num_features, out_features= 2)]) # Add the new layer with outputting 2 categories
model.classifier = nn.Sequential(*classifier_layers) # Replace the model classifier, Overwriting the original

VGG16 = binary_VGG16_transfer_learning(
    model = model, 
    train_dataloader = train_dataloader, 
    test_dataloader = test_dataloader,
    criterion = nn.CrossEntropyLoss(), 
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.classifier.parameters()), lr=0.001, momentum=0.9),
    writer = SummaryWriter(log_dir=log_dir),
    device = device
)

final_stats = VGG16.model_training(numOfEpoch = 100)
print(final_stats)

# Save the model.
os.makedirs(log_dir, exist_ok=True)
VGG16.save(log_dir + 'most_recent_model_dict.pt')

# Release resources.
VGG16.close()
