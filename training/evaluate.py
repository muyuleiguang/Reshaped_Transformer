import time
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from metrics.metrics import (
    classification_metrics, regression_metrics, average_dtw
)

def evaluate_model(model, dataloader, device, log_dir=None, plot_examples=3):
    """
    在验证/测试集上评估模型，并可选地记录日志、绘制对比图。
    - log_dir: 如果不为 None，会把指标写入 log_dir/metrics.txt，并保存示例图
    - plot_examples: 从数据集中抽几个样本画 真值 vs 预测图
    Returns dict of metrics.
    """
    model.eval()
    all_cls_pred, all_cls_true = [], []
    all_reg_pred, all_reg_true = [], []

    start_time = time.time()
    with torch.no_grad():
        for X, y_cls, y_reg in dataloader:
            X = X.to(device)
            cls_logits, y_pred_reg = model(X)
            cls_pred = torch.argmax(cls_logits, dim=-1).cpu().numpy()

            all_cls_true.append(y_cls.numpy())
            all_cls_pred.append(cls_pred)
            all_reg_true.append(y_reg.numpy())
            all_reg_pred.append(y_pred_reg.cpu().numpy())
    elapsed = time.time() - start_time

    # 合并
    all_cls_true = np.concatenate(all_cls_true, axis=0)
    all_cls_pred = np.concatenate(all_cls_pred, axis=0)
    all_reg_true = np.vstack(all_reg_true)
    all_reg_pred = np.vstack(all_reg_pred)

    # 计算指标
    cls_m = classification_metrics(all_cls_true, all_cls_pred)
    reg_m = regression_metrics(all_reg_true, all_reg_pred)
    dtw = average_dtw(all_reg_true, all_reg_pred)

    results = {
        'cls_accuracy': cls_m['accuracy'],
        'cls_precision': cls_m['precision'],
        'cls_recall': cls_m['recall'],
        'cls_f1': cls_m['f1'],
        'reg_mse': reg_m['mse'],
        'reg_mae': reg_m['mae'],
        'reg_dtw': dtw,
        'eval_time_s': elapsed
    }

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        # 写指标到 txt
        with open(os.path.join(log_dir, 'metrics.txt'), 'a') as f:
            f.write(time.asctime() + '\n')
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
            f.write('\n')

        # 绘制部分示例对比图
        for i in range(min(plot_examples, len(all_reg_true))):
            plt.figure(figsize=(6,3))
            plt.plot(all_reg_true[i], label='True')
            plt.plot(all_reg_pred[i], label='Pred')
            plt.legend()
            plt.title(f"Sample {i} True vs Pred")
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f"sample_{i}.png"))
            plt.close()

    return results
# training/evaluate.py

import time
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from metrics.metrics import (
    classification_metrics, regression_metrics, average_dtw
)

def evaluate_model(model, dataloader, device, log_dir=None, plot_examples=3):
    """
    在验证/测试集上评估模型，并可选地记录日志、绘制对比图。
    - log_dir: 如果不为 None，会把指标写入 log_dir/metrics.txt，并保存示例图
    - plot_examples: 从数据集中抽几个样本画 真值 vs 预测图
    Returns dict of metrics.
    """
    model.eval()
    all_cls_pred, all_cls_true = [], []
    all_reg_pred, all_reg_true = [], []

    start_time = time.time()
    with torch.no_grad():
        for X, y_cls, y_reg in dataloader:
            X = X.to(device)
            cls_logits, y_pred_reg = model(X)
            cls_pred = torch.argmax(cls_logits, dim=-1).cpu().numpy()

            all_cls_true.append(y_cls.numpy())
            all_cls_pred.append(cls_pred)
            all_reg_true.append(y_reg.numpy())
            all_reg_pred.append(y_pred_reg.cpu().numpy())
    elapsed = time.time() - start_time

    # 合并
    all_cls_true = np.concatenate(all_cls_true, axis=0)
    all_cls_pred = np.concatenate(all_cls_pred, axis=0)
    all_reg_true = np.vstack(all_reg_true)
    all_reg_pred = np.vstack(all_reg_pred)

    # 计算指标
    cls_m = classification_metrics(all_cls_true, all_cls_pred)
    reg_m = regression_metrics(all_reg_true, all_reg_pred)
    dtw = average_dtw(all_reg_true, all_reg_pred)

    results = {
        'cls_accuracy': cls_m['accuracy'],
        'cls_precision': cls_m['precision'],
        'cls_recall': cls_m['recall'],
        'cls_f1': cls_m['f1'],
        'reg_mse': reg_m['mse'],
        'reg_mae': reg_m['mae'],
        'reg_dtw': dtw,
        'eval_time_s': elapsed
    }

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        # 写指标到 txt
        with open(os.path.join(log_dir, 'metrics.txt'), 'a') as f:
            f.write(time.asctime() + '\n')
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
            f.write('\n')

        # 绘制部分示例对比图
        for i in range(min(plot_examples, len(all_reg_true))):
            plt.figure(figsize=(6,3))
            plt.plot(all_reg_true[i], label='True')
            plt.plot(all_reg_pred[i], label='Pred')
            plt.legend()
            plt.title(f"Sample {i} True vs Pred")
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f"sample_{i}.png"))
            plt.close()

    return results
