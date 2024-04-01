import logging
import os


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def create_logger(dataset, model_name,log_name):
    # if not os.path.exists('./newlogs/{}/'.format(dataset)):
    #     os.makedirs('./newlogs/{}/'.format(dataset))

    if dataset == 0:
        assert False
    elif dataset == "GD":
        file_logger = setup_logger(model_name,'./logs/GD/{}_{}_{}.log'.format(model_name, dataset,log_name))
    elif dataset == "HSS":
        file_logger = setup_logger(model_name,'./logs/GD/{}_{}_{}.log'.format(model_name, dataset,log_name))
    elif dataset == "TD":
        file_logger = setup_logger(model_name,'./logs/GD/{}_{}_{}.log'.format(model_name, dataset,log_name))
    elif "Yahoo" in dataset :
        file_logger = setup_logger(model_name, './logs/GD/{}_{}_{}.log'.format(model_name, dataset,log_name))
    elif "ECG" in dataset :
        file_logger = setup_logger(model_name, './logs/ECG/{}_{}_{}.log'.format(model_name, dataset,log_name))
    return file_logger


def display(logger, metrics_result):
    logger.info('============================')
    # logger.info('avg_TN = {}'.format(metrics_result.TN))
    # logger.info('avg_FP = {}'.format(metrics_result.FP))
    # logger.info('avg_FN = {}'.format(metrics_result.FN))
    # logger.info('avg_TP = {}'.format(metrics_result.TP))
    # logger.info('avg_precision = {}'.format(metrics_result.precision))
    # logger.info('avg_recall = {}'.format(metrics_result.recall))
    # logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
    logger.info('last_roc_auc = {}'.format(metrics_result.roc_auc))
    logger.info('last_pr_auc = {}'.format(metrics_result.pr_auc))
    logger.info('last_F1 = {}'.format(metrics_result.F1))
    logger.info('avg_roc_auc = {}'.format(metrics_result.avg_roc_auc))
    logger.info('avg_pr_auc = {}'.format(metrics_result.avg_pr_auc))
    logger.info('avg_F1 = {}'.format(metrics_result.avg_F1))
    logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
    logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
    logger.info('best_F1 = {}'.format(metrics_result.best_F1))
    # logger.info('avg_cks = {}'.format(metrics_result.cks))
    # logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
    # logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
    # logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
    # logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
    # logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
    # logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
    # logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
    # logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
    # logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
    # logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
    # logger.info('training_time = {}'.format(metrics_result.training_time))
    # logger.info('testing_time = {}'.format(metrics_result.testing_time))
    # logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
    logger.info('============================')