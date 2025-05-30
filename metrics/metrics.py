# 性能度量模块：定义用于评估分类准确率、精确率、召回率、F1-score，及回归误差的函数
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from fastdtw import fastdtw  # 使用 fastdtw 库计算动态时间规整距离

def classification_accuracy(y_pred_labels, y_true_labels):
    y_pred_labels = np.array(y_pred_labels)
    y_true_labels = np.array(y_true_labels)
    return (y_pred_labels == y_true_labels).sum() / len(y_true_labels)

def classification_precision(y_pred_labels, y_true_labels):
    return precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)

def classification_recall(y_pred_labels, y_true_labels):
    return recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)

def classification_f1(y_pred_labels, y_true_labels):
    return f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)

def mean_squared_error_metric(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return float(np.mean((y_pred - y_true) ** 2))

def mean_absolute_error_metric(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return float(np.mean(np.abs(y_pred - y_true)))

def dynamic_time_warping(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    # fastdtw 返回 (距离, 路径)
    distance, _ = fastdtw(y_pred, y_true)
    return float(distance)

def compute_metrics(outputs, targets):
    """
    给定模型输出与标签，统一计算分类与回归指标。
    参数:
        outputs: 模型输出 (class_logits, regression_output)
        targets: 标签 (y_class, y_reg)
    返回:
        包含 accuracy, precision, recall, f1, mse, mae, dtw 的字典
    """
    class_logits, reg_output = outputs
    y_class, y_reg = targets

    pred_labels = class_logits.argmax(dim=1).cpu().numpy()
    true_labels = y_class.cpu().numpy()

    pred_reg = reg_output.view(-1).detach().cpu().numpy()
    true_reg = y_reg.cpu().numpy()

    return {
        'accuracy': classification_accuracy(pred_labels, true_labels),
        'precision': classification_precision(pred_labels, true_labels),
        'recall': classification_recall(pred_labels, true_labels),
        'f1_score': classification_f1(pred_labels, true_labels),
        'mse': mean_squared_error_metric(pred_reg, true_reg),
        'mae': mean_absolute_error_metric(pred_reg, true_reg),
        'dtw': dynamic_time_warping(pred_reg, true_reg)
    }
