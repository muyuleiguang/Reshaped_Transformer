# evaluate.py

import torch
from training.loss import compute_loss
from metrics.metrics import (
    classification_accuracy,
    classification_precision,
    classification_recall,
    classification_f1,
    mean_squared_error_metric,
    mean_absolute_error_metric,
    dynamic_time_warping
)

def evaluate(model, data_loader):
    """
    在验证集或测试集上评估多任务模型的性能，返回包含损失和多种指标的字典。
    Args:
        model: 训练好的多任务模型
        data_loader: torch.utils.data.DataLoader，提供 (X, y_class, y_reg) 批次
    Returns:
        dict 包含键：
            'loss_total'：总损失（分类+回归）
            'loss_class'：分类损失
            'loss_reg'： 回归损失
            'accuracy'：分类准确率
            'precision'：分类精确率（Macro）
            'recall'：分类召回率（Macro）
            'f1_score'：分类 F1 分数（Macro）
            'mse'：回归均方误差
            'mae'：回归平均绝对误差
            'dtw'： 预测序列与真实序列的 DTW 距离
    """
    model.eval()  # 切换到评估模式（关闭 dropout 等）
    device = next(model.parameters()).device  # 获取模型所在设备

    # 以下用于累计损失和样本数量
    total_class_loss = 0
    total_reg_loss   = 0
    total_samples    = 0

    # 以下列表用于收集所有批次的预测和真实值，以计算指标
    all_pred_labels = []
    all_true_labels = []
    all_pred_regs   = []
    all_true_regs   = []

    with torch.no_grad():  # 评估时不计算梯度
        for X, y_class, y_reg in data_loader:
            # 将数据移动到同一设备
            X, y_class, y_reg = X.to(device), y_class.to(device), y_reg.to(device)

            # 模型前向，outputs=(class_logits, reg_out)
            outputs = model(X)

            # 调用统一的损失计算接口，返回 (total_loss, class_loss, reg_loss)
            _, class_loss, reg_loss = compute_loss(outputs, (y_class, y_reg))

            batch = X.size(0)
            total_class_loss += class_loss.item() * batch
            total_reg_loss   += reg_loss.item()   * batch
            total_samples    += batch

            # 拆分分类和回归输出
            class_logits, reg_out = outputs

            # 分类：取 argmax 得到标签预测
            preds = class_logits.argmax(dim=1)
            all_pred_labels.extend(preds.cpu().tolist())
            all_true_labels.extend(y_class.cpu().tolist())

            # 回归：将 reg_out 展平后收集
            all_pred_regs.extend(reg_out.view(-1).cpu().tolist())
            flat_true = y_reg.reshape(-1).cpu().tolist()
            all_true_regs.extend(flat_true)

    # 计算平均损失
    avg_class_loss = total_class_loss / total_samples
    avg_reg_loss   = total_reg_loss   / total_samples
    avg_total_loss = avg_class_loss + avg_reg_loss

    # 计算分类指标
    accuracy  = classification_accuracy(all_pred_labels, all_true_labels)
    precision = classification_precision(all_pred_labels, all_true_labels)
    recall    = classification_recall(all_pred_labels, all_true_labels)
    f1_score  = classification_f1(all_pred_labels, all_true_labels)

    # 计算回归指标
    mse      = mean_squared_error_metric(all_pred_regs, all_true_regs)
    mae      = mean_absolute_error_metric(all_pred_regs, all_true_regs)
    dtw_dist = dynamic_time_warping(all_pred_regs, all_true_regs)

    # 返回一个包含所有指标的字典
    return {
        'loss_total': avg_total_loss,
        'loss_class': avg_class_loss,
        'loss_reg':   avg_reg_loss,
        'accuracy':   accuracy,
        'precision':  precision,
        'recall':     recall,
        'f1_score':   f1_score,
        'mse':        mse,
        'mae':        mae,
        'dtw':        dtw_dist
    }
