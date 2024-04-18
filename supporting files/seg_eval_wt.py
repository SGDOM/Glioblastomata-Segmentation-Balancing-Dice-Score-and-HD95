import numpy as np
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

def metric_class(pred, label):
    """
    Calcula metricas (accuracy,especificidad y sensibilidad) para un ejemplo 
    en particular del WT(tumor completo)

    Parametros:
        pred (np.array): Array binario de predicciones(num classes, height, width, depth).
        label (np.array): Array binario de etqieutas(num classes, height, width, depth).

    Returns:
        accuracy(float): accuracy
        sensitivity (float): precision 
        specificity (float): recall 
    """

    # verdadero positivo
    tp = np.sum((pred == 1) & (label == 1))

    # verdadero negativo
    tn = np.sum((pred == 0) & (label == 0))
    
    # falso positivo
    fp = np.sum((pred == 1) & (label == 0))
    
    # falso negativo
    fn = np.sum((pred == 0) & (label == 1))

    # accuracy,sensibilidad,especificidad,coeficiente de dice
    accuracy = (tp+tn)/ (tp + tn + fp + fn)
    sensitivity = (tp)/ (tp + fn)
    specificity = (tn)/ (tn + fp)
    dice = (2*tp)/ ((2*tp)+fp+fn)
    hd=BraTS_HD95(pred,label)

    return accuracy,sensitivity, specificity,dice,hd