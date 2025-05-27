import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def classification_metrics(y_true, y_pred, average='macro'):
    """
    计算分类任务的评估指标。
    :param y_true: 真实标签，形状为 (n_samples,)，整数类型
    :param y_pred: 预测标签，形状为 (n_samples,)，整数类型
    :param average: 平均方式 ('macro', 'micro', 'weighted', 'samples')，默认 'macro'
    :return: 包含准确率、精确率、召回率和 F1 分数的字典
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("y_true and y_pred must be NumPy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    except Exception as e:
        logger.exception("Error computing classification metrics: %s", e)
        acc = prec = rec = f1 = float('nan')
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def regression_metrics(y_true, y_pred):
    """
    计算回归任务的评估指标。
    :param y_true: 真实值，形状为 (n_samples, seq_len) 或 (n_samples,)
    :param y_pred: 预测值，形状与 y_true 相同
    :return: 包含均方误差 (MSE) 和平均绝对误差 (MAE) 的字典
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("y_true and y_pred must be NumPy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    try:
        y_t = np.ravel(y_true)
        y_p = np.ravel(y_pred)
        mse = mean_squared_error(y_t, y_p)
        mae = mean_absolute_error(y_t, y_p)
    except Exception as e:
        logger.exception("Error computing regression metrics: %s", e)
        mse = mae = float('nan')
    return {
        'mse': mse,
        'mae': mae
    }

def dtw_distance(y_true, y_pred):
    """
    计算单条序列的 DTW 距离。
    :param y_true: 真实序列，形状为 (seq_len,)
    :param y_pred: 预测序列，形状为 (seq_len,)
    :return: DTW 距离
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("y_true and y_pred must be NumPy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    try:
        distance, _ = fastdtw(y_true, y_pred, dist=euclidean)
    except Exception as e:
        logger.exception("Error computing DTW distance for one sequence: %s", e)
        distance = float('nan')
    return distance

def average_dtw(y_true_batch, y_pred_batch, max_dtw_samples=1000):
    """
    计算一批序列的平均 DTW 距离。
    :param y_true_batch: 真实序列批次，形状为 (n_samples, seq_len)
    :param y_pred_batch: 预测序列批次，形状为 (n_samples, seq_len)
    :return: 平均 DTW 距离
    """
    if not isinstance(y_true_batch, np.ndarray) or not isinstance(y_pred_batch, np.ndarray):
        raise ValueError("y_true_batch and y_pred_batch must be NumPy arrays")
    if y_true_batch.shape != y_pred_batch.shape:
        raise ValueError("y_true_batch and y_pred_batch must have the same shape")
        # 采样逻辑：如果样本数超过 max_dtw_samples，则随机选择样本
    if len(y_true_batch) > max_dtw_samples:
        indices = np.random.choice(len(y_true_batch), max_dtw_samples, replace=False)
        y_true_batch = y_true_batch[indices]
        y_pred_batch = y_pred_batch[indices]
    dists = []
    for yt, yp in zip(y_true_batch, y_pred_batch):
        dists.append(dtw_distance(yt, yp))
    return float(np.nanmean(dists))
