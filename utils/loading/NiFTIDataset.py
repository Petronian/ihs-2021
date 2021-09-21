# pylint: disable=no-name-in-module
# pylint erroneously fails to recognize function from_numpy in torch.

from torch import from_numpy
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

class NiFTIDataset(Dataset):
    """
    Dataset responsible for reading in a metadata CSV or DataFrame containing
    information in the following format:

    Parameters
    ----------
    metadata:     CSV or DataFrame containing the metadata information.
    root:         Base filepath used by the relative filepaths contained
                  within the metadata file.
    transform:    Pytorch transforms dynamically applied to the loaded
                  images within the Dataset. Optional.

    Metadata Columns
    -------
    NIFTI_Name:   Name of the image filepath.
    NIFTI_Path:   Relative path to the image from the parent path, parent path
                  being the directory 'studies' folder is at.
    Label:        Label of the image given as a number.
    Label_Folder: Name of the folder containing images with the given label.
    S25_Path:     Relative filepath of the 25th layer of the NiFTI image.

    Output
    ------
    Data are output from the DataLoader as a Python dictionary with the following
    key-value pairs.

    image:        A key containing a single Tensor containing all of the images
                  loaded in the current batch.
    label:        A Tensor containing all of the labels for the current batch
                  of images.

    Corresponding image information is located at the same relative position
    within each value of every key-value pair. In other words, the image data
    and label for an image at index X are at index X within both the image
    and label Tensor.

    Created by Peter Lais on 09/21/2021.
    """

    def __init__(self, metadata, root, transform=None):
        # Check if root exists.
        if(not os.path.isdir(root)):
            exit('ImageDataset: Root does not exist.')

        # Load metadata (path_to_csv or dataframe).
        if (type(metadata) == pd.core.frame.DataFrame):
            metadata_df = metadata.copy()
        else:
            metadata_df = pd.read_csv(metadata)

        # Optional transforms on top of tensor-ization.
        self.transform = transform

        # Metadata attribute
        self.metadata = metadata_df
        # Image directory
        self.root = root  

    def __len__(self):
        # Number of rows of metadata dataframe.
        return len(self.metadata)

    def __getitem__(self, idx):
        # Extract relevant information.
        image_row = self.metadata.iloc[idx]
        print(image_row)
        label = image_row['Label'] 
        image_path = os.path.join(self.root, image_row['S25_Path'])
        
        # Load image into numpy array and convert to Tensor.
        image = from_numpy(np.load(image_path))
        
        # Custom transforms.
        if self.transform:
            image = self.transform(image)
        
        # Return image and label.
        return {'image': image, 'label': label}