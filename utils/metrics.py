import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, confusion_matrix


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
    cm = confusion_matrix(y_true=labels, y_pred=(predicts >= 0.5).astype(int)).tolist()

    return {'average_precision': average_precision, 'roc_auc': roc_auc, 'roc': roc, 'cm': cm}

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
