#!/usr/bin/env python
# coding: utf-8

# # Create Metadata for Images
# 
# Zitian Tang\
# 9/20/2021

# In[29]:


import numpy as np
import pandas as pd

## Filesystem ##
import os
from os import listdir
from os.path import isfile, join

## PyTorch and TorchVision ##
import torch
import torchvision

## Nibabel ##
import nibabel as nib


# # User Variables

# In[14]:


## Image Directory ##
image_directory = '../data/CT_images/COVID19_1110/studies'
CT0_dir = os.path.join(image_directory, 'CT-0')
CT1_dir = os.path.join(image_directory, 'CT-1')
CT2_dir = os.path.join(image_directory, 'CT-2')
CT3_dir = os.path.join(image_directory, 'CT-3')
CT4_dir = os.path.join(image_directory, 'CT-4')

## Filelists ##
## CT-0 ##
CT0_list = [f for f in listdir(CT0_dir) if isfile(join(CT0_dir,f))]
## CT-1 ##
CT1_list = [f for f in listdir(CT1_dir) if isfile(join(CT1_dir,f))]
## CT-2 ##
CT2_list = [f for f in listdir(CT2_dir) if isfile(join(CT2_dir,f))]
## CT-3 ##
CT3_list = [f for f in listdir(CT3_dir) if isfile(join(CT3_dir,f))]
## CT-4 ##
CT4_list = [f for f in listdir(CT4_dir) if isfile(join(CT4_dir,f))]

## nth Slice Specified ##
slice_num = 25
new_folder_path = '../data/CT_images/COVID_Zitian_Danni/Slice_25'

## Output: Metadata csv with NIFTI names, Labels, and relative paths ##
metadata_path = 'metadata.csv'


# # Save Specified slice to new Folder

# In[30]:


## Metadata dataframe ##
metadata = pd.read_csv(metadata_path, index_col=0)

## Loop through and save nth slice ##
for i in range(metadata.shape[0]):
    # get image path to load image in
    image_row = metadata.iloc[i]
    image_path = os.path.join(image_directory, image_row['NIFTI_Path'])
    # generate new filename for images
    image_name = image_row['NIFTI_Name']
    image_name = image_name.split('.')[0] + '_Slice_25.npy'
    # load image from path
    data = nib.load(image_path)
    data = data.get_fdata()
    # get nth slice
    current_slice = data[:,:,slice_num]
    # saving image
    full_path = new_folder_path + '/' + image_name
    np.save(full_path, current_slice)


# # Generate Dataframe

# In[8]:


## NIFTI Name (ID) ##
IDs = CT0_list + CT1_list + CT2_list + CT3_list + CT4_list

## NIFTI Path (Relative paths from image_directory) ##
rpath_0 = [join('CT-0', img) for img in CT0_list]
rpath_1 = [join('CT-1', img) for img in CT1_list]
rpath_2 = [join('CT-2', img) for img in CT2_list]
rpath_3 = [join('CT-3', img) for img in CT3_list]
rpath_4 = [join('CT-4', img) for img in CT4_list]
relative_paths = rpath_0 + rpath_1 + rpath_2 + rpath_3 + rpath_4

## Labels ##
labels = [int(f[3]) for f in relative_paths]

## Label folder ##
label_folder = [f[:4] for f in relative_paths]

## Convert Lists to Arrays ##
IDs = np.expand_dims(IDs, axis=-1)
relative_paths = np.expand_dims(relative_paths, axis=-1)
labels = np.expand_dims(labels, axis=-1)
label_folder = np.expand_dims(label_folder, axis=-1)

## Print Save Location ##
print('Metadata Path is: %s'% metadata_path)

## Create Dataframe ##
metadata_df = pd.DataFrame(np.hstack((IDs, relative_paths, labels, 
                                      label_folder)),
                           columns = ['NIFTI_Name', 'NIFTI_Path', 'Label',
                                      'Label_Folder'])
metadata_df.head()


# In[41]:


# Add the Saved Slice Path to Dataframe Column
# slice_list = listdir(new_folder_path)

slice_list = [join(new_folder_path,f) for f in listdir(new_folder_path) 
              if isfile(join(new_folder_path,f))]
column_name = 'S%s_Path'%f'{slice_num}'
metadata_df[column_name] = slice_list

metadata_df.head()


# In[42]:


## Save Metadata CSV ##
metadata_df.to_csv(metadata_path)


# In[ ]:




