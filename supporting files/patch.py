import cv2
import h5py
import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image
from keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
)
from keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1.logging import INFO, set_verbosity

set_verbosity(INFO)

K.set_image_data_format("channels_first")


def get_sub_volume(image, label, 
                   orig_x = 240, orig_y = 240, orig_z = 155, 
                   output_x = 160, output_y = 160, output_z = 16,
                   num_classes = 4, max_tries = 1000, 
                   background_threshold=0.95):
    """
    Extraer subvolumenes de las imágenes originales

    Parametros:
        image (np.array): imagen original, 
            de tamaño (orig_x, orig_y, orig_z, num_channels)
        label (np.array): label original. 
            labels coded using discrete values rather than
            a separate dimension, 
            so this is of shape (orig_x, orig_y, orig_z)
        orig_x (int): x_dim de imagen de entrada
        orig_y (int): y_dim de imagen de entrada
        orig_z (int): z_dim de imagen de entrada
        output_x (int): x_dim deseado de salida
        output_y (int): y_dim deseado de salida
        output_z (int): z_dim deseado de salida
        num_classes (int): numero de clases de labels
        max_tries (int): maximo numero de iteraciones para extraer muestra
        background_threshold (float): limite del subvolumen que tiene una fraccion de 
        fondo

    returns:
        X (np.array): subvolumen de IMR original
            (num_channels, output_x, output_y, output_z)
        y (np.array): etiqueta correspondiente a X 
            (num_classes, output_x, output_y, output_z)
    """
    # Inicializando
    X = None
    y = None
    
    tries = 0
    
    while tries < max_tries:
        # Muestrear aleatoriamente el subvolumen
        start_x = np.random.randint(0, orig_x - output_x+1)
        start_y = np.random.randint(0, orig_y - output_y+1)
        start_z = np.random.randint(0, orig_z - output_z+1)

        # Extaemos area relevante de la máscara
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        
        # One-hot encoding de categorías.
        # Agregamos 4 dimensiones (numero de clases es la etiquetas de las mascaras)
        # (output_x, output_y, output_z, num_classes)
        y = keras.utils.to_categorical(y, num_classes=num_classes)

        # Relación de fondo
        # Se escoge el indice 0 ya que aqui esta el tumor completo (WT)
        bgrd_ratio = np.sum(y[:, :, :, 0])/(output_x * output_y * output_z)

        # Incrementando contador
        tries += 1

        # Como máximo el 95% del subvolumen deben ser regiones tumorales
        if bgrd_ratio < background_threshold:

            # Copiamos subvolumen a X
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])
            
            # Dimension actual de X: (x_dim, y_dim, z_dim, num_channels)
            # Dimension de x debe ser: (num_channels, x_dim, y_dim, z_dim)
            X = np.moveaxis(X, 3, 0)

            # Dimension actual de y: (x_dim, y_dim, z_dim, num_classes)
            # Dimension de y debe ser: (num_classes, x_dim, y_dim, z_dim)
            y = np.moveaxis(y, 3, 0)

            
            # Excluimos el indice 1 donde se encuentra la máscara del WT
            y = y[1:, :, :, :]

    
            return start_x,start_y,start_z,X, y

    print(f"No se encontro subvolumen en {tries} iteraciones. Probar con otro registro...")