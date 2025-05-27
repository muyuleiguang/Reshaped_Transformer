import torch
import torch.nn as nn

class MultiTaskUncertaintyLoss(nn.Module):
    """
    多任务损失函数，结合分类和回归损失，使用不确定性加权。
    :param num_classes: 分类任务的类别数（如4）
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.log_sigma_cls = nn.Parameter(torch.zeros(1))
        self.log_sigma_reg = nn.Parameter(torch.zeros(1))

    def forward(self, y_cls_pred, y_cls_true, y_reg_pred, y_reg_true):
        loss_cls = self.classification_loss(y_cls_pred, y_cls_true)
        loss_reg = self.regression_loss(y_reg_pred, y_reg_true)
        sigma_cls = torch.exp(self.log_sigma_cls)
        sigma_reg = torch.exp(self.log_sigma_reg)
        total_loss = (loss_cls / (2 * sigma_cls**2) + torch.log(sigma_cls)) + \
                     (loss_reg / (2 * sigma_reg**2) + torch.log(sigma_reg))
        return total_loss