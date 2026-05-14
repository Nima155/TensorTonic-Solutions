import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    classes = np.unique(y_true)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    conf_matrix = np.zeros((classes.shape[0], 4))
    class_to_indx = {}
    for i, c in enumerate(classes):
        conf_matrix[i, 0] = ((y_pred == c) & (y_true == c)).sum()
        conf_matrix[i, 1] = ((y_pred == c) & (y_true != c)).sum()
        conf_matrix[i, 2] = ((y_pred != c) & (y_true == c)).sum()
        conf_matrix[i, 3] = (y_true == c).sum() / y_pred.size
        class_to_indx[c] = i
        
    if average == "micro":
        return {
            "accuracy": (conf_matrix[:, 0].sum()) / (y_pred.size),
            "recall": (conf_matrix[:, 0].sum()) / (conf_matrix[:, 0].sum() + conf_matrix[:, 2].sum()),
            "precision": (conf_matrix[:, 0].sum()) / (conf_matrix[:, 0].sum() + conf_matrix[:, 1].sum()),
            "f1": 2 * conf_matrix[:,0].sum() / (2 * conf_matrix[:, 0].sum() + conf_matrix[:, 1].sum() + conf_matrix[:, 2].sum()) 
        }
    elif average == "macro":
        rec = conf_matrix[:, 0] / (conf_matrix[:, 0] + conf_matrix[:, 2])
        prec = conf_matrix[:, 0] / (conf_matrix[:, 0] + conf_matrix[:, 1])
        f1 = 2 * conf_matrix[:,0] / (2 * conf_matrix[:, 0] + conf_matrix[:, 1] + conf_matrix[:, 2])
        return {
            "accuracy": (conf_matrix[:, 0].sum()) / (y_pred.size),
            "recall": rec.mean(),
            "precision": prec.mean(),
            "f1": f1.mean()
        }
    elif average == "weighted":
        weight = conf_matrix[:, 3]
        rec = conf_matrix[:, 0] * weight / (conf_matrix[:, 0] + conf_matrix[:, 2])
        prec = conf_matrix[:, 0] * weight / (conf_matrix[:, 0] + conf_matrix[:, 1])
        f1 = 2 * conf_matrix[:,0] * weight / (2 * conf_matrix[:, 0] + conf_matrix[:, 1] + conf_matrix[:, 2])
        return {
            "accuracy": (conf_matrix[:, 0].sum()) / (y_pred.size),
            "recall": (rec).sum(),
            "precision": (prec).sum(),
            "f1": (f1 ).sum()
        }
    elif average == "binary":
        rec = conf_matrix[class_to_indx[pos_label], 0] / (conf_matrix[class_to_indx[pos_label], 0] + conf_matrix[class_to_indx[pos_label], 2])
        
        prec = conf_matrix[class_to_indx[pos_label], 0] / (conf_matrix[class_to_indx[pos_label], 0] + conf_matrix[class_to_indx[pos_label], 1])

        f1 = 2 * conf_matrix[class_to_indx[pos_label],0]  / (2 * conf_matrix[class_to_indx[pos_label], 0] + conf_matrix[class_to_indx[pos_label], 1] + conf_matrix[class_to_indx[pos_label], 2])
        return {
            "accuracy": (conf_matrix[:, 0].sum()) / (y_pred.size),
            "recall": rec,
            "precision": prec,
            "f1": f1
        }
    return 