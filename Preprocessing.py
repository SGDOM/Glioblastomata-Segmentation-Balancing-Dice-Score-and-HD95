

!pip show tensorflow

import tensorflow as tf
tf.test.gpu_device_name()
print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

print("GPU:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(physical_devices))

!pip install tensorflow==2.8.0

pip install MedPy

import zipfile
dataset_path = "/content/drive/My Drive/spektral-master.zip"  # Replace with your dataset path
zfile = zipfile.ZipFile(dataset_path)
zfile.extractall()

import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import SimpleITK as sitk
from skimage import io
from medpy.metric import dc, hd95
import glob
import nibabel as nib
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import os
#from layers import GraphCNN
#from spektral.layers import GraphConv
from scipy import sparse as sp

from google.colab import drive
drive.mount('/content/drive')
drive.mount("/content/drive", force_remount=True)

PATH_TRAIN = "/content/drive/MyDrive/BraTS-2021/brats2021@100"
PATH_VALI= "/content/drive/MyDrive/BraTS-2021/vali"
PATH= "/content/drive/MyDrive/BraTS-2021/DatasetFinal/imagesTr/train/"
PATH1="/content/drive/MyDrive/BraTS-2021/DatasetFinal/labelsTr/train/"
PATH2= "/content/drive/MyDrive/BraTS-2021/DatasetFinal/imagesTr/valid/"
PATH3= "/content/drive/MyDrive/BraTS-2021/DatasetFinal/labelsTr/valid/"

"""## **A)Train**"""

os.chdir(PATH_TRAIN)
list_train = os.listdir(PATH_TRAIN)

list_train

for i in range(len(list_train)):
  list_sec = os.listdir("./" + list_train[i])

  image_path_flair = list_sec[0]
  image_path_t1 = list_sec[2]
  image_path_seg = list_sec[1]
  image_path_t1ce = list_sec[3]
  image_path_t2 = list_sec[4]

  image_obj_flair = nib.load(PATH_TRAIN + "/"  + str(list_train[i]) + "/" + str(image_path_flair))
  image_obj_t1ce = nib.load(PATH_TRAIN + "/"  + str(list_train[i]) + "/" + str(image_path_t1ce))
  image_obj_t1 = nib.load(PATH_TRAIN + "/"  + str(list_train[i]) + "/" + str(image_path_t1))
  image_obj_t2 = nib.load(PATH_TRAIN + "/"  + str(list_train[i]) + "/" + str(image_path_t2))
  image_obj_seg = nib.load(PATH_TRAIN + "/"  + str(list_train[i]) + "/" + str(image_path_seg))

  image_data_flair = image_obj_flair.get_fdata()
  image_data_t1ce = image_obj_t1ce.get_fdata()
  image_data_t1 = image_obj_t1.get_fdata()
  image_data_t2 = image_obj_t2.get_fdata()
  image_path_seg = image_obj_seg.get_fdata()


  image_path_seg[image_path_seg == 4.] = 3.

  image_path_seg =nib.Nifti1Image(image_path_seg, np.eye(4))

  total_BraTS21 = np.stack((image_data_flair,
                  image_data_t1ce,
                  image_data_t1,
                  image_data_t2),
                  axis = 3)
  total_BraTS21 = nib.Nifti1Image(total_BraTS21, np.eye(4))

  nib.save(total_BraTS21,
           os.path.join(PATH + "/" , 'BraTS21_Training_'+ str(i+1).zfill(3) + '_total.nii')
           )

  nib.save(image_path_seg,
           os.path.join(PATH1 + "/", 'BraTS21_Training_'+ str(i+1).zfill(3) + '_segf.nii')
           )

!ls

image_path_flair = './BraTS2021_00051/BraTS2021_00051_flair.nii.gz'
image_path_t1ce = './BraTS2021_00051/BraTS2021_00051_t1ce.nii.gz'
image_path_t1 = './BraTS2021_00051/BraTS2021_00051_t1.nii.gz'
image_path_t2 = './BraTS2021_00051/BraTS2021_00051_t2.nii.gz'

image_obj_flair = nib.load(image_path_flair)
image_obj_t1ce = nib.load(image_path_t1ce)
image_obj_t1 = nib.load(image_path_t1)
image_obj_t2 = nib.load(image_path_t2)

image_data_flair = image_obj_flair.get_fdata()
image_data_t1ce = image_obj_t1ce.get_fdata()
image_data_t1 = image_obj_t1.get_fdata()
image_data_t2 = image_obj_t2.get_fdata()

type(image_data_flair)

image_path_total = '/content/drive/MyDrive/BraTS-2021/DatasetFinal/imagesTr/train/BraTS21_Training_003_total.nii'
image_obj_total = nib.load(image_path_total)
image_data_total = image_obj_total.get_fdata()

total_BraTS21_Training_001 = np.stack((image_data_flair,
                                        image_data_t1ce,
                                        image_data_t1,
                                        image_data_t2),
                                        axis = 3)

total_BraTS21_Training_001.shape

image_data_total.shape

(total_BraTS21_Training_001 == image_data_total).all()

image_path_segf = '/content/drive/MyDrive//BraTS-2021/DatasetFinal/labelsTr/train/BraTS21_Training_003_segf.nii'
image_obj_segf = nib.load(image_path_segf)
image_data_segf = image_obj_segf.get_fdata()

image_data_segf.shape

# Segmentacion inicial
image_path_seg = './BraTS2021_00051/BraTS2021_00051_seg.nii.gz'
image_obj_seg = nib.load(image_path_seg)
image_data_seg = image_obj_seg.get_fdata()

image_data_seg.shape

np.unique(image_data_seg)


image_data_seg[image_data_seg == 4.] = 3.

np.unique(image_data_seg)


(image_data_seg == image_data_segf).all()


"""## **B)Validation**"""

os.chdir(PATH_VALI)
list_vali = os.listdir(PATH_VALI)

list_vali[250:251]

for i in range(len(list_vali)):
  list_sec = os.listdir("./" + list_vali[i])

  image_path_flair = list_sec[0]
  image_path_t1 = list_sec[2]
  image_path_seg = list_sec[1]
  image_path_t1ce = list_sec[3]
  image_path_t2 = list_sec[4]


  image_obj_flair = nib.load(PATH_VALI + "/"  + str(list_vali[i]) + "/" + str(image_path_flair))
  image_obj_t1ce = nib.load(PATH_VALI + "/"  + str(list_vali[i]) + "/" + str(image_path_t1ce))
  image_obj_t1 = nib.load(PATH_VALI + "/"  + str(list_vali[i]) + "/" + str(image_path_t1))
  image_obj_t2 = nib.load(PATH_VALI + "/"  + str(list_vali[i]) + "/" + str(image_path_t2))
  image_obj_seg = nib.load(PATH_VALI + "/"  + str(list_vali[i]) + "/" + str(image_path_seg))

  image_data_flair = image_obj_flair.get_fdata()
  image_data_t1ce = image_obj_t1ce.get_fdata()
  image_data_t1 = image_obj_t1.get_fdata()
  image_data_t2 = image_obj_t2.get_fdata()
  image_path_seg = image_obj_seg.get_fdata()

  image_path_seg[image_path_seg == 4.] = 3.

  image_path_seg =nib.Nifti1Image(image_path_seg, np.eye(4))

  # Juntamos las 4 secuencias separadas en una solo arreglo
  total_BraTS21 = np.stack((image_data_flair,
                  image_data_t1ce,
                  image_data_t1,
                  image_data_t2),
                  axis = 3)
  total_BraTS21 = nib.Nifti1Image(total_BraTS21, np.eye(4))

  nib.save(total_BraTS21,
           os.path.join(PATH2 + "/", 'BraTS21_Validation_'+ str(i+1).zfill(3) + '_total.nii')
           )
  nib.save(image_path_seg,
           os.path.join(PATH3 + "/", 'BraTS21_Validation_'+ str(i+1).zfill(3) + '_segf.nii')
           )

image_path_flair = './BraTS2021_01416/BraTS2021_01416_flair.nii.gz'
image_path_t1ce = './BraTS2021_01416/BraTS2021_01416_t1ce.nii.gz'
image_path_t1 = './BraTS2021_01416/BraTS2021_01416_t1.nii.gz'
image_path_t2 = './BraTS2021_01416/BraTS2021_01416_t2.nii.gz'

image_obj_flair = nib.load(image_path_flair)
image_obj_t1ce = nib.load(image_path_t1ce)
image_obj_t1 = nib.load(image_path_t1)
image_obj_t2 = nib.load(image_path_t2)

image_data_flair = image_obj_flair.get_fdata()
image_data_t1ce = image_obj_t1ce.get_fdata()
image_data_t1 = image_obj_t1.get_fdata()
image_data_t2 = image_obj_t2.get_fdata()

type(image_data_flair)

total_BraTS21_Validation_001 = np.stack((image_data_flair,
                                        image_data_t1ce,
                                        image_data_t1,
                                        image_data_t2),
                                        axis = 3)

total_BraTS21_Validation_001.shape

image_path_total = '/content/drive/MyDrive/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/DatasetFinal/imagesTr/valid/BraTS21_Validation_001_total.nii'
image_obj_total = nib.load(image_path_total)
image_data_total = image_obj_total.get_fdata()
image_data_total.shape

(image_data_total == total_BraTS21_Validation_001).all()


import os
import sys
PATH_ORIGEN = "/content/drive/MyDrive/BraTS-2020"
os.chdir(PATH_ORIGEN)
sys.path.append(os.path.abspath(PATH_ORIGEN))

!pip install keras==2.4

import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import patch
import visualization
import util

HOME_DIR = "./DatasetFinal/"
DATA_DIR = HOME_DIR

def load_case(image_nifty_file, label_nifty_file):
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())

    return image, label

image, label = load_case(DATA_DIR + "imagesTr/train/BraTS21_Training_011_total.nii", DATA_DIR + "labelsTr/train/BraTS21_Training_011_segf.nii")
image = util.get_labeled_image(image, label)
util.plot_image_grid(image)

util.visualize_data_gif(util.get_labeled_image(image, label))

def get_sub_volume(image, label,
                   orig_x = 240, orig_y = 240, orig_z = 155,
                   output_x = 160, output_y = 160, output_z = 16,
                   num_classes = 4, max_tries = 1000,
                   background_threshold=0.98):
    """
    Extract random sub-volume from original images.

    Args:
        image (np.array): original image,
            of shape (orig_x, orig_y, orig_z, num_channels)
        label (np.array): original label.
            labels coded using discrete values rather than
            a separate dimension,
            so this is of shape (orig_x, orig_y, orig_z)
        orig_x (int): x_dim of input image
        orig_y (int): y_dim of input image
        orig_z (int): z_dim of input image
        output_x (int): desired x_dim of output
        output_y (int): desired y_dim of output
        output_z (int): desired z_dim of output
        num_classes (int): number of class labels
        max_tries (int): maximum trials to do when sampling
        background_threshold (float): limit on the fraction
            of the sample which can be the background

    returns:
        X (np.array): sample of original image of dimension
            (num_channels, output_x, output_y, output_z)
        y (np.array): labels which correspond to X, of dimension
            (num_classes, output_x, output_y, output_z)
    """
    # Initialize features and labels with `None`
    X = None
    y = None

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    tries = 0

    while tries < max_tries:
        # randomly sample sub-volume by sampling the corner voxel
        # hint: make sure to leave enough room for the output dimensions!
        start_x = np.random.randint(0, orig_x - output_x+1)
        start_y = np.random.randint(0, orig_y - output_y+1)
        start_z = np.random.randint(0, orig_z - output_z+1)

        # extract relevant area of label
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]

        # One-hot encode the categories.
        # This adds a 4th dimension, 'num_classes'
        # (output_x, output_y, output_z, num_classes)
        y = tf.keras.utils.to_categorical(y,num_classes=num_classes)

        # compute the background ratio
        bgrd_ratio = np.sum(y[:, :, :, 0])/(output_x * output_y * output_z)

        # increment tries counter
        tries += 1

        # if background ratio is below the desired threshold,
        # use that sub-volume.
        # otherwise continue the loop and try another random sub-volume
        if bgrd_ratio < background_threshold:

            # make copy of the sub-volume
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])

            # change dimension of X
            # from (x_dim, y_dim, z_dim, num_channels)
            # to (num_channels, x_dim, y_dim, z_dim)
            X = np.moveaxis(X, 3, 0)

            # change dimension of y
            # from (x_dim, y_dim, z_dim, num_classes)
            # to (num_classes, x_dim, y_dim, z_dim)
            #y = np.moveaxis(y, 3, 0)

            ### END CODE HERE ###

            # take a subset of y that excludes the background class
            # in the 'num_classes' dimension
            #y = y[:, :, :,1:]

            return start_x,start_y,start_z,X, y

    # if we've tried max_tries number of samples
    # Give up in order to avoid looping forever.
    print(f"Tried {tries} times to find a sub-volume. Giving up...")

visualization.visualize_patch(x[1, :, :, :], y[:,:,:,1])

def standardize(image):

    # Inicializamos matriz de ceros
    standardized_image = np.zeros(image.shape)

    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            # canal c y dimension z
            image_slice = image[c,:,:,z]

            # Estandarizamos
            centered = image_slice - np.mean(image_slice)
            if np.std(centered) != 0:
                centered_scaled = centered / np.std(centered)

            # Guardamos la imagen estandarizada
            standardized_image[c, :, :, z] = centered_scaled

    return standardized_image

X_norm = standardize(x)
print(f"stddv para X_norm[0, :, :, 0]: {X_norm[0,:,:,0].std():.2f}")

visualization.visualize_patch(X_norm[0, :, :, :], y[:,:,:,1])

h5f = h5py.File("/content/drive/MyDrive/BraTS-2021/DatasetFinal/processed/valid/BraTS21_Training_003_total.nii_16_x25_y29_z100.h5", 'r')
h5f.keys()
x=h5f['x']
y=h5f['y']

x.shape

y.shape

PATH_FIN_TRAIN = "/content/drive/MyDrive/BraTS-2021/DatasetFinal/processed/train/"
PATH_FIN_VALID = "/content/drive/MyDrive/BraTS-2021/DatasetFinal/processed/valid/"

list_train = os.listdir(PATH_FIN_VALID)
#list_label= os.listdir(PATH1)
#list_valid = os.listdir(PATH2)
#list_vlabel= os.listdir(PATH3)

list_t=os.listdir("/content/drive/MyDrive/Task01_BrainTumour/Task01_BrainTumour/DatasetFinal/processed/valid/")
len(list_t)

list_train

for i in range(len(list_train)):
  count = 0
  while count < 20:
   count = count + 1
   image, label = load_case("/content/drive/MyDrive/BraTS-2021/DatasetFinal/imagesTr/train/"+list_train[i],"/content/drive/MyDrive/BraTS-2021/DatasetFinal/labelsTr/train/"+list_label[i])
   start_x, start_y, start_z, X, y = get_sub_volume(image, label)
   X_norm = standardize(X)
   with h5py.File(PATH_FIN_TRAIN + 'BraTS21_Training1_'+ str(i+1).zfill(3) + '_total.nii' + "_" + str(count) + "_" +
               "x" + str(start_x) + "_" + "y" + str(start_y) + "_" + "z" + str(start_z) + ".h5", "w") as hdf:
     hdf.create_dataset('x', data = X_norm)
     hdf.create_dataset('y', data = y)

for i in range(len(list_valid)):
  count = 0
  while count < 20:
   count = count + 1
   image, label = load_case("/content/drive/MyDrive/BraTS-2021/DatasetFinal/imagesTr/valid/"+list_valid[i],"/content/drive/MyDrive/BraTS-2021/DatasetFinal/labelsTr/valid/"+list_vlabel[i])
   start_x, start_y, start_z, X, y = get_sub_volume(image, label)
   X_norm = standardize(X)
   with h5py.File(PATH_FIN_VALID + 'BraTS21_Training_'+ str(i+1).zfill(3) + '_total.nii' + "_" + str(count) + "_" +
               "x" + str(start_x) + "_" + "y" + str(start_y) + "_" + "z" + str(start_z) + ".h5", "w") as hdf:
     hdf.create_dataset('x', data = X_norm)
     hdf.create_dataset('y', data = y)

data = {}
data['train'] = []
data['valid'] =[]
for i in os.listdir(PATH_FIN_TRAIN):
  data['train'].append(i)

for j in os.listdir(PATH_FIN_VALID):
  data['valid'].append(j)

with open ('./DatasetFinal/processed/config.json','w') as file:
  json.dump(data,file)



