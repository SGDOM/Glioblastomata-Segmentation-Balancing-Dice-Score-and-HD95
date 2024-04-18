import numpy as np
import pandas as pd
from medpy.metric import dc, hd95
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

def metric_class(pred, label, class_num):
    """
    Calcula metricas (accuracy,especificidad y sensibilidad) para un ejemplo 
    en particular de una clase determinada

    Parametros:
        pred (np.array): Array binario de predicciones(num classes, height, width, depth).
        label (np.array): Array binario de etqieutas(num classes, height, width, depth).
        class_num (int): [0 - (num_classes -1)] indica para que clase(edema,TC,ET) se va
        a calcular las métricas

    Returns:
        accuracy(float): accuracy
        sensitivity (float): precision 
        specificity (float): recall 
    """

    # extraemos submatriz para la clase especificada
    class_pred = pred[class_num]
    class_label = label[class_num]
    
    # verdadero positivo
    tp = np.sum((class_pred == 1) & (class_label == 1))

    # verdadero negativo
    tn = np.sum((class_pred == 0) & (class_label == 0))
    
    # falso positivo
    fp = np.sum((class_pred == 1) & (class_label == 0))
    
    # falso negativo
    fn = np.sum((class_pred == 0) & (class_label == 1))

    # accuract,sensibilidad,especificidad,coeficiente de dice
    accuracy = (tp+tn)/(tp + tn + fp + fn)
    sensitivity = (tp) / (tp + fn)
    specificity = (tn) / (tn + fp)
    dice = (2*tp) /((2*tp)+fp+fn)
    hd=BraTS_HD95(pred,label)

    return accuracy,sensitivity, specificity,dice,hd
    
    
def get_metrics_df(pred, label):
    patch_metrics = pd.DataFrame(
        columns = ['Nucleo del tumor', 
                   'Edema peritumoral', 
                   'Tumor realzado'], 
        index = ['Accuracy',
                 'Sensitivity',
                 'Specificity',
                 'Dice'])
    
    for i, class_name in enumerate(patch_metrics.columns):
        acc,sens, spec,dice = metric_class(pred, label, i)
        patch_metrics.loc['Accuracy', class_name] = round(acc,4)
        patch_metrics.loc['Sensitivity', class_name] = round(sens,4)
        patch_metrics.loc['Specificity', class_name] = round(spec,4)
        patch_metrics.loc['Dice', class_name] = round(dice,4)

    return patch_metrics