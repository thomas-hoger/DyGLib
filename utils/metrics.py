import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, confusion_matrix
import torch.nn.functional as F


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    fpr, tpr, threshold = roc_curve(y_true=labels, y_score=predicts)
    roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "threshold": threshold.tolist()}
    
    tn, fp, fn, tp = cm = confusion_matrix(y_true=labels, y_pred=(predicts >= 0.5).astype(int)).ravel().tolist()
    positives = tp + fn
    negatives = tn + fp 
    tpr = tp/positives
    fpr = fp/negatives

    return {'average_precision': average_precision, 'tpr': tpr, 'fpr': fpr, 'roc_auc': roc_auc, 'roc': roc, 'cm': cm}

def get_autoencoder_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the autoencoder task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    fpr, tpr, threshold = roc_curve(y_true=labels, y_score=predicts)
    roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "threshold": threshold.tolist()}
    cm = confusion_matrix(y_true=labels, y_pred=(predicts >= 0.5).astype(int)).tolist()
    
    tn, fp, fn, tp = cm = confusion_matrix(y_true=labels, y_pred=(predicts >= 0.5).astype(int)).ravel().tolist()
    positives = tp + fn
    negatives = tn + fp 
    tpr = tp/positives
    fpr = fp/negatives

    return {'average_precision': average_precision, 'tpr': tpr, 'fpr': fpr, 'roc_auc': roc_auc, 'roc': roc, 'cm': cm}

def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
