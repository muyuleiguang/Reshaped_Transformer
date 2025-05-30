# 损失函数模块：定义分类和回归任务的损失函数
import torch.nn as nn

# 定义分类和回归的基础损失
classification_criterion = nn.CrossEntropyLoss()
regression_criterion   = nn.MSELoss()

def multi_task_loss(class_logits, class_labels, reg_output, reg_labels, alpha=1.0):
    """
    计算分类和回归的联合损失。
    参数:
        class_logits: 分类任务的预测logits张量 [batch_size, num_classes]
        class_labels: 分类任务的真实标签张量 [batch_size]
        reg_output: 回归任务的预测输出张量 [batch_size] 或 [batch_size, 1]
        reg_labels: 回归任务的真实值张量 [batch_size]
        alpha: 回归损失项的权重系数（可根据需要调整）。默认值为1.0。
    返回:
        total_loss: 分类损失和回归损失的加权和（标量）
        class_loss: 分类损失值（标量）
        reg_loss: 回归损失值（标量）
    """
    # 计算分类损失
    class_loss = classification_criterion(class_logits, class_labels)
    # 直接计算矩阵 MSE，reg_output 和 reg_labels 都是 [batch, Lpred]
    reg_loss = regression_criterion(reg_output, reg_labels)

    # 按权重alpha将两者相加得到总损失
    total_loss = class_loss + alpha * reg_loss
    return total_loss, class_loss, reg_loss

def compute_loss(outputs, targets, alpha=1.0):
    """
    对外统一接口：给定模型输出和标签，返回总损失、分类损失、回归损失
    """
    class_logits, reg_output = outputs
    class_labels, reg_labels = targets
    return multi_task_loss(class_logits, class_labels, reg_output, reg_labels, alpha)

# 损失函数模块：定义分类和回归任务的损失函数
import torch.nn as nn

# 分类损失函数（交叉熵）
classification_criterion = nn.CrossEntropyLoss()
# 回归损失函数（均方误差）
regression_criterion = nn.MSELoss()

def multi_task_loss(class_logits, class_labels, reg_output, reg_labels, alpha=1.0):
    """
    计算多任务模型的总损失（分类+回归）。
    参数：
        class_logits: 分类输出 [batch_size, num_classes]
        class_labels: 分类标签 [batch_size]
        reg_output: 回归输出 [batch_size] 或 [batch_size, 1]
        reg_labels: 回归标签 [batch_size]
        alpha: 回归损失的权重系数（默认1.0）
    返回：
        total_loss: 总损失
        class_loss: 分类损失
        reg_loss: 回归损失
    """
    class_loss = classification_criterion(class_logits, class_labels)
    reg_loss = regression_criterion(reg_output, reg_labels)
    total_loss = class_loss + alpha * reg_loss
    return total_loss, class_loss, reg_loss

def compute_loss(outputs, targets, alpha=1.0):
    """
    与模型接口统一的损失计算函数，方便train.py中调用。
    参数：
        outputs: 模型输出(class_logits, reg_output)
        targets: 标签(class_labels, reg_labels)
        alpha: 回归损失权重
    返回：
        total_loss, class_loss, reg_loss
    """
    class_logits, reg_output = outputs
    class_labels, reg_labels = targets
    return multi_task_loss(class_logits, class_labels, reg_output, reg_labels, alpha)
