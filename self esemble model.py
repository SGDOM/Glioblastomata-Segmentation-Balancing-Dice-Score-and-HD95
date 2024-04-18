import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, regularizers
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,Dropout
)
from tensorflow.keras.layers import concatenate,Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), 
                   epsilon=0.00001):


    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))


    return dice_loss

def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), 
                     epsilon=0.00001):
    

    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean((dice_numerator)/(dice_denominator))
    

    return dice_coefficient

def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv3D(filters=inter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv3D(filters=inter_shape,
                     kernel_size=3,
                     strides=(2,2,2),
                     padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg =concatenate([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv3D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling3D(
        size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3],
              shape_x[4] // shape_sigmoid[4]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    
    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = layers.multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv3D(filters=shape_x[1],
                    kernel_size=1,
                    strides=1,
                    padding='same')(attn_coefficients)
    output = layers.BatchNormalization()(output)
    return output

def attention_block(F_g, F_l, F_int):
            g = Conv3D(F_int, 1, padding="same")(F_g)
            g = layers.BatchNormalization()(g)

            x = Conv3D(F_int, 1, padding="same")(F_l)
            x = layers.BatchNormalization()(x)

            psi = Add()([g, x])
            psi = Activation("relu")(psi)

            psi = Conv3D(1, 1, padding="same")(psi)
            psi = layers.BatchNormalization()(psi)
            psi = Activation("sigmoid")(psi)

            return layers.Multiply()([F_l, psi])


def unet_model_3d(loss_function, input_shape=(4, 160, 160, 16),
                  pool_size=(2, 2, 2), n_labels=3,
                  initial_learning_rate=0.0001,
                  deconvolution=False,metrics=[],
                  activation_name="sigmoid"):
    
    #################### PASO 1: CAPA DE ENTRADA #############################
     
    inputs = Input(input_shape)
    
    conv1 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv11 = concatenate([conv1,inputs], axis = 1)
    conv1 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv12 = concatenate([inputs,conv1, conv11], axis = 1)
    conv1 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv1 = concatenate([inputs,conv1, conv12], axis = 1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv21 = concatenate([conv2,pool1], axis = 1)
    conv2 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    conv22 = concatenate([pool1,conv21, conv2], axis = 1)
    conv2 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    conv2 = concatenate([pool1,conv2, conv22], axis = 1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv31 = concatenate([conv3,pool2], axis = 1)
    conv3 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv31)
    conv32 = concatenate([pool2,conv31, conv3], axis = 1)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv32)
    pool3 = Dropout(0.25)(pool3)
    conv3 = Conv3D(512, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv3 = Conv3D(512, (3, 3, 3),  activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    #drop5 = Dropout(0.5)(conv5)
    
    #################### FIN CAPA INTERMEDIA #########################
    
    #################### PASO 4: INICIO DECODER #########################
    
    #### Bloque 1 ####
    
    conv6 = AttnGatingBlock(conv32, conv3, 256)
    up6=UpSampling3D(size=(2,2,2))(conv3)
    merge6 = concatenate([conv6,up6], axis = 1)
    conv6 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    out1 = Conv3D(256, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up6=UpSampling3D(size=(2,2,2))(out1)
    up6 = Conv3D(256, (1, 1, 1), activation = activation_name, padding = 'same', kernel_initializer = 'he_normal')(up6)
    #### Bloque 2 ####
    
    conv7 = AttnGatingBlock(conv2, out1, 128)
    up7=UpSampling3D(size=(2,2,2))(out1)
    merge7 = concatenate([conv7,up7], axis = 1)
    conv7 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    out2=Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    out22 = concatenate([out2,up6], axis = 1)
    up7=UpSampling3D(size=(2,2,2))(out22)
    up7 = Conv3D(128, (1, 1, 1), activation = activation_name, padding = 'same', kernel_initializer = 'he_normal')(up7)
    #### Bloque 3 ####
    
  
    conv8 = AttnGatingBlock(conv1, out2, 64)
    up8=UpSampling3D(size=(2,2,2))(out2)
    merge8 = concatenate([conv8,up8], axis = 1)
    conv8 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv3D(64, (1, 1, 1), activation = activation_name, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8=concatenate([conv8,up7],axis = 1)
    
    #################### FIN DECODER #########################
     
     
    #### Bloque 4 ####
    #up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #merge9 = concatenate([conv1,up9], axis = 3)
    #conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    #conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    ################### PASO 5: CAPA DE CLASIFICACION ################
    
    conv9 = Conv3D(n_labels, (1, 1, 1), activation = activation_name, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)
    
    ################### PASO 6: CREAMOS EL OBJETO MODELO ################
    
    model = Model(inputs, conv9)
    
    ################### PASO 7: COMPILAMOS EL MODELO ################

    model.compile(optimizer = Adam(lr = initial_learning_rate), loss = loss_function, metrics = metrics)
    
    return model
