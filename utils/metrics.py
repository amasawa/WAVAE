import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, \
    roc_curve, auc, cohen_kappa_score
from torch.autograd import Variable

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN

def create_label_based_on_quantile(score, quantile=99):
    # higher scores is more likely to be anomalies
    label = np.full(score.shape[0], 1)
    threshold = np.percentile(score, quantile)
    label[score > threshold] = -1
    return label
def create_label_based_on_zscore(zscore, threshold, sign=False):
    label = np.full(zscore.shape[0], 1)
    if not sign:
        label[zscore > threshold] = -1
        label[zscore < -threshold] = -1
    else:
        label[zscore > threshold] = -1
    # label[abs(zscore) > abs(threshold)] = -1
    return label
def zscore(error):
    '''
    Calculate z-score using error
    :param error: error time series
    :return: z-score
    '''
    mu = np.nanmean(error)
    gamma = np.nanstd(error)
    z_score = (error - mu) / gamma
    return z_score

class MetricsResult(object):
    def __init__(self, TN=None, FP=None, FN=None, TP=None, precision=None, recall=None, fbeta=None, pr_auc=None,
                 roc_auc=None, cks=None, best_TN=None, best_FP=None, best_FN=None, best_TP=None, best_precision=None,
                 best_recall=None, best_fbeta=None, best_pr_auc=None, best_roc_auc=None, best_cks=None,
                 training_time=None, testing_time=None, memory_estimation=None,avg_pr_auc=None,avg_roc_auc=None,F1=None,best_F1=None,avg_F1=None):
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.TP = TP

        self.precision = precision
        self.recall = recall
        self.fbeta = fbeta
        self.pr_auc = pr_auc
        self.roc_auc = roc_auc
        self.cks = cks
        self.F1 = F1
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks
        self.best_F1 = best_F1

        self.training_time = training_time
        self.testing_time = testing_time
        self.memory_estimation = memory_estimation

        self.avg_pr_auc = avg_pr_auc
        self.avg_roc_auc = avg_roc_auc
        self.avg_F1 = avg_F1