#!/usr/bin/env python
# coding: utf-8

# # Create Metadata for Images
# 
# Zitian Tang\
# 9/20/2021

# In[2]:


import numpy as np
import pandas as pd

# Filesystem
import os
from os import listdir
from os.path import isfile, join


# # User Variables

# In[3]:


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

## Output: Metadata csv with NIFTI names, Labels, and relative paths ##
metadata_path = 'metadata.csv'


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


# In[ ]:


## Add 25th Slice Path to Dataframe Column ##


# In[9]:


## Save Metadata CSV ##
metadata_df.to_csv(metadata_path)

