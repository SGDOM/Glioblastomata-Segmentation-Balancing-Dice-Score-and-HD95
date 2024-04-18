import os
import sys
PATH_ORIGEN = "/content/drive/MyDrive/BraTS-2020"
os.chdir(PATH_ORIGEN)
PATH_METRICS = "./DatasetFinal/processed/train/"
PATH_VAL = "./DatasetFinal/processed/valid/"
sys.path.append(os.path.abspath(PATH_ORIGEN))

import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import h5py
import scipy
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint,CSVLogger
import visualization
import patch
import model1
import model2
import model14
import generate
import preprocessing
import metrics1
import metrics
import seg_eval_wt

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3),
                   epsilon=0.00001):


    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))

    return dice_loss


ALPHA = 0.3
BETA = 0.7
GAMMA = 1

from keras import backend as K
import tensorflow as tf

def FocalTverskyLoss(y_true, y_pred, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):

        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)

        #True Positives, False Positives & False Negatives
        TP = K.sum((y_pred * y_true))
        FP = K.sum(((1-y_true) * y_pred))
        FN = K.sum((y_true * (1-y_pred)))

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = K.pow((1 - Tversky), gamma)

        return FocalTversky



ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75
def ASYMFocalTverskyLoss(y_true, y_pred, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):

        #flatten label and prediction tensors
        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)

        #True Positives, False Positives & False Negatives
        TP = K.sum((y_pred * y_true))
        FP = K.sum(((1-y_true) * y_pred))
        FN = K.sum((y_true * (1-y_pred)))

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        #FocalTversky = K.pow((1 - Tversky), gamma)

        back_dice = (1-Tversky)
        fore_dice = (1-Tversky) * K.pow(1-Tversky, -gamma)

        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))

        return loss

def generalised_dice_loss_3d(Y_gt, Y_pred):
    smooth = 1e-5
    w = tf.reduce_sum(Y_gt, axis=[1, 2, 3])
    w = 1 / (w ** 2 + smooth)

    numerator = Y_gt * Y_pred
    numerator = w * tf.reduce_sum(numerator, axis=[1, 2, 3])
    numerator = tf.reduce_sum(numerator, axis=1)

    denominator = Y_pred + Y_gt
    denominator = w * tf.reduce_sum(denominator, axis=[1, 2, 3])
    denominator = tf.reduce_sum(denominator, axis=1)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = tf.reduce_mean(1 - gen_dice_coef)
    return loss

def FDICE(y_true, y_pred):
    # Obtain Soft DSC
    #dice = soft_dice_loss(y_true, y_pred)
    #dice=generalised_dice_loss_3d(y_true, y_pred)
    # Obtain Crossentropy
    #Focal=FocalTverskyLoss(y_true, y_pred, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6)
    # Return sum
    #averaged_mask = K.pool3d(y_true, pool_size=(11, 11, 11), strides=(1, 1, 1), padding='same', pool_mode='avg')
    #border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    #weight = K.ones_like(averaged_mask)
    #w0 = K.sum(weight)
    #weight += border * 2
    #w1 = K.sum(weight)
    #weight *= (w0 / w1)
    loss =  generalised_dice_loss_3d(y_true, y_pred)+FocalTverskyLoss(y_true, y_pred, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6)
    return loss

def dice_coefficient(y_true, y_pred, axis=(1, 2, 3),
                     epsilon=0.00001):


    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean((dice_numerator)/(dice_denominator))

    return dice_coefficient

def BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref)





model = model1.unet_model_3d(loss_function=FocalTverskyLoss, metrics=[dice_coefficient])

model.summary()

from tensorflow.keras.utils import  plot_model
plot_model(model,to_file='unet.png',show_shapes=True)

base_dir = PATH_ORIGEN + "/DatasetFinal/processed/"

with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)

PATH_WEIGHTS = "/content/drive/MyDrive/UNet3D2/"

model_checkpoint = ModelCheckpoint(filepath = PATH_WEIGHTS + "model-{epoch:04d}-{val_loss:.4f}.hdf5",
                                   monitor = 'val_loss',
                                   verbose = 1,
                                   save_best_only = False,
                                   #save_weights_only=False,
                                   save_freq = 'epoch'
                                   )
# History
csv_logger = CSVLogger(PATH_WEIGHTS + "model_history_log.csv", append=True)

train_generator = generate.VolumeDataGenerator(config["train"],
                                           base_dir + "train/",
                                           batch_size=1,
                                           dim=(160, 160, 16),
                                           verbose=0)
valid_generator = generate.VolumeDataGenerator(config["valid"],
                                            base_dir + "valid/",
                                            batch_size=1,
                                            dim=(160, 160, 16),
                                            verbose=0)

steps_per_epoch = 
n_epochs = 
validation_steps = 

history = model.fit_generator(generator=train_generator,
                steps_per_epoch = steps_per_epoch,
                epochs = n_epochs,
                callbacks=[model_checkpoint,csv_logger],
                use_multiprocessing=True,
                validation_data=valid_generator,
                validation_steps=validation_steps)

PATH_HISTORY = "/content/drive/MyDrive/UNet3D/model_history_log (1).csv"
history = pd.read_csv(PATH_HISTORY)

import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(12, 8)

# plot loss
plt.subplot(2,2,1)
plt.title('Soft Dice Loss')
plt.xlabel("epoch")
plt.plot(history['loss'], color='blue', label='train')
plt.plot(history['val_loss'],color ='orange',label='validation')
plt.legend(['entrenamiento', 'validación'], loc='upper left')

# plot dice coefficient
plt.subplot(2,2,2)
plt.title('Dice coefficient')
plt.xlabel("epoch")
plt.plot(history['dice_coefficient'], color='blue', label='train')
plt.plot(history['val_dice_coefficient'], color='orange', label='validation')
plt.legend(['entrenamiento', 'validación'], loc='upper left')
plt.show()

checkpoint_filepath ="/content/drive/MyDrive/UNet3D/model-0003-0.3547.hdf5"

!nvidia-smi:

model.load_weights(checkpoint_filepath)

#model1=tf.keras.models.load_model("/content/drive/MyDrive/UNet3D/model-0001-0.3268.hdf5", compile=False)
model2=tf.keras.models.load_model("/content/drive/MyDrive/UNet3D1/model-0002-0.4722.hdf5", compile=False)
model3=tf.keras.models.load_model("/content/drive/MyDrive/UNet3D1/model-0001-0.3003.hdf5", compile=False)
#model4=tf.keras.models.load_model("/content/drive/MyDrive/UNet3D1/model-0001-0.1918.hdf5", compile=False)
#model5=tf.keras.models.load_model("/content/drive/MyDrive/UNet3D1/model-0001-0.1843.hdf5", compile=False)

model=tf.keras.models.load_model("/content/drive/MyDrive/UNet3D/model-0001-0.3268.hdf5", compile=False)
model.compile(optimizer = Adam(lr = 0.000001), loss =  FDICE, metrics=[dice_coefficient])

from tensorflow.keras.optimizers import Adam

#model1.compile(optimizer = Adam(learning_rate = 0.000001), loss =  FDICE, metrics=[dice_coefficient])
model2.compile(optimizer = Adam(learning_rate = 0.000001), loss =  FDICE, metrics=[dice_coefficient])
model3.compile(optimizer = Adam(learning_rate = 0.000001), loss =  FDICE, metrics=[dice_coefficient])
#model4.compile(optimizer = Adam(learning_rate = 0.000001), loss =  FocalTverskyLoss, metrics=[dice_coefficient])
#model5.compile(optimizer = Adam(learning_rate = 0.000001), loss =  FocalTverskyLoss, metrics=[dice_coefficient])

val_loss, val_dice = model.evaluate_generator(valid_generator)
print(f"validation soft dice loss: {val_loss:.4f}")
print(f"validation dice coefficient: {val_dice:.4f}")

image, label = preprocessing.load_case("/content/drive/MyDrive/BraTS-2021/DatasetFinal/imagesTr/train/BraTS21_Training_048_total.nii",
                         "/content/drive/MyDrive/BraTS-2021/DatasetFinal/labelsTr/train/BraTS21_Training_048_segf.nii")

start_x,start_y,start_z,X,y = preprocessing.get_sub_volume(image,label)

y.shape

G=y

X_norm = preprocessing.standardize(X)

visualization.visualize_patch(X_norm[0, :, :, :], y[0])
print(np.unique(y[0],return_counts=True))

pred = util.predict_and_viz(image, label, model, .5, loc=(120, 120, 70))

X_norm_with_batch_dimension = np.expand_dims(X_norm, axis=0)
print(X_norm_with_batch_dimension.shape)

patch_pred = model.predict(X_norm_with_batch_dimension)


import time
start_time = time.time()
model2_predictions = model2.predict(X_norm_with_batch_dimension)
model3_predictions = model3.predict(X_norm_with_batch_dimension)
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")



threshold = 0.5
patch_pred[patch_pred > threshold] = 1.0  # tumor class
patch_pred[patch_pred <= threshold] = 0.0 # no tumor

num_pixels = np.prod(patch_pred[0,2, :, :, :].shape)
num_pixels

volume = units * pow(1, 3)
volume

#model1_predictions = model1.predict(X_norm_with_batch_dimension)
model2_predictions = model2.predict(X_norm_with_batch_dimension)
model3_predictions = model3.predict(X_norm_with_batch_dimension)
#model4_predictions = model4.predict(X_norm_with_batch_dimension)
#np.save("model1_predictions .npy", model1_predictions)
#np.save("model2_predictions .npy", model2_predictions)
#np.save("model3_predictions .npy", model3_predictions)
    #patch_pred4 = model4.predict(X_norm_with_batch_dimension)
    #patch_pred5 = model5.predict(X_norm_with_batch_dimension)

import numpy as np

# Define a list of model predictions (binary masks) from different models
model_predictions = []
#model_predictions.append(model1_predictions)
model_predictions.append(model2_predictions)
model_predictions.append(model3_predictions)
#model_predictions.append(model4_predictions)
# Add predictions from as many models as needed

# Perform ensemble averaging (you can also use other ensemble methods)
ensemble_prediction = np.mean(model_predictions, axis=0)

# Binarize the ensemble prediction using a threshold
threshold = 0.5  # Adjust as needed
ensemble_prediction_binary = (ensemble_prediction > threshold).astype(np.uint8)

# Save the ensemble prediction to a file or use it for further analysis
#np.save("ensemble_prediction.npy", ensemble_prediction_binary)

import numpy as np

# Define a list of model predictions (binary masks) from different models
model_predictions = []
#model_predictions.append(model1_predictions)
model_predictions.append(model2_predictions)
model_predictions.append(model3_predictions)
#model_predictions.append(model4_predictions)
# Add predictions from as many models as needed

# Stack the model predictions along a new axis (axis 0)
stacked_predictions = np.stack(model_predictions, axis=0)

# Perform majority voting ensemble prediction
ensemble_prediction = np.median(stacked_predictions, axis=0)

# Binarize the ensemble prediction using a threshold
threshold = 0.5  # Adjust as needed
ensemble_prediction_binary = (ensemble_prediction > threshold).astype(np.uint8)


# Save the ensemble prediction to a file or use it for further analysis
#np.save("majority_voting_ensemble_prediction.npy", ensemble_prediction_binary)

visualization.visualize_patch(X_norm[0, :, :, :], ensemble_prediction_binary [0,2, :, :, :])
print(np.unique(y[0],return_counts=True))

accuracy_TC,precision_TC,recall_TC,dice_TC = metrics.metric_class(ensemble_prediction_binary[0],y,0) # tumor core
accuracy_ED,precision_ED,recall_ED,dice_ED = metrics.metric_class(ensemble_prediction_binary[0],y,1) # Edema
accuracy_ET,precision_ET,recall_ET,dice_ET = metrics.metric_class(ensemble_prediction_binary[0],y,2)  # enhanced tumor
hd95_core=BraTS_HD95(ensemble_prediction_binary[0][0,:,:,:],y[0,:,:,:])
hd95_Edema=BraTS_HD95(ensemble_prediction_binary[0][1,:,:,:],y[1,:,:,:])
hd95_ET=BraTS_HD95(ensemble_prediction_binary[0][2,:,:,:],y[2,:,:,:])

ensemble_prediction_binary.shape

# Find the mask y_WT as the sum of the other masks
y_WT = y[0] + y[1] + y[2]
y_WT = np.where(y_WT >= 1, 1, 0)

# Find the prediction of the mask y_WT as the sum of the other masks
patch_pred_WT = ensemble_prediction_binary[0,0, :, :, :] + ensemble_prediction_binary[0,1, :, :, :] + ensemble_prediction_binary[0,2, :, :, :]
patch_pred_WT = np.where(patch_pred_WT >= 1, 1, 0)

y_Core=y[0] + y[2]
y_Core = np.where(y_Core >= 1, 1, 0)

patch_pred_core =ensemble_prediction_binary[0,0, :, :, :] + ensemble_prediction_binary[0,2, :, :, :]
patch_pred_core= np.where(patch_pred_core >= 1, 1,0)

a1,b1,c1,d1,e1 = seg_eval_wt.metric_class(patch_pred_core,y_Core) # Accuracy, sensitivity, specificity and dice for TC
    #a2,b2,c2,d2 = metrics.metric_class(patch_pred[0],y,1) # Accuracy, sensitivity, specificity and dice for ED
a2,b2,c2,d2,e2 = seg_eval_wt.metric_class(patch_pred_WT,y_WT) # Accuracy, sensitivity, specificity and dice for WT
a3,b3,c3,d3,e3 = metrics1.metric_class(ensemble_prediction_binary[0],y,2) #

acc = [a1,a2,a3]
sens =[b1,b2,b3]
spec =[c1,c2,c3]
dice =[d1,d2,d3]
hd=[e1,e2,e3]

print(a1)
print(a2)
print(a3)
print(b1)
print(b2)
print(b3)
print(c1)
print(c2)
print(c3)
print(d1)
print(d2)
print(d3)
print(e1)
print(e2)
print(e3)

metrics_final = pd.DataFrame(columns = ['Tumor Core', 'Whole Tumor', 'Enhanced Tumor'], index = ['Sensitivity','specificity','Dice','hd'])

metrics_final
