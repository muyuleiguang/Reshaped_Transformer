import torch
from training.evaluate import evaluate_model
from loss import MultiTaskUncertaintyLoss
# 注意: MultiTaskUncertaintyLoss、evaluate_model 等自定义模块需在其他处定义或导入

def train(train_loader, val_loader, model, num_epochs=100, learning_rate=1e-4,
          patience=10, scheduler_patience=5, weight_decay=0):
    """
    训练多任务模型，支持同时进行分类和回归任务。
    
    参数:
        train_loader: 训练数据加载器 (每个 batch 返回 inputs, cls_labels, reg_labels)
        val_loader: 验证数据加载器 (格式同上)
        model: PyTorch 模型，应输出分类预测和回归预测 (返回 (cls_output, reg_output))
        num_epochs: 最大训练轮数
        learning_rate: 学习率 (Adam 优化器)
        patience: EarlyStopping 的耐心值 (验证集 loss 持续不下降时提前终止)
        scheduler_patience: ReduceLROnPlateau 调度器的耐心值
        weight_decay: 权重衰减 (L2)
    返回:
        best_model_path: 保存的最佳模型权重文件路径
        history: 训练日志列表 (每轮 epoch 的指标字典)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = MultiTaskUncertaintyLoss()  # 不确定性加权损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience)

    best_loss = float('inf')
    early_stop_counter = 0
    best_model_path = "best_model.pt"
    history = []

    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0.0
        # 训练一个 epoch
        for inputs, cls_labels, reg_labels in train_loader:
            inputs, cls_labels, reg_labels = inputs.to(device), cls_labels.to(device), reg_labels.to(device)

            optimizer.zero_grad()
            cls_output, reg_output = model(inputs)
            loss = criterion(cls_output, cls_labels, reg_output, reg_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, cls_labels, reg_labels in val_loader:
                inputs, cls_labels, reg_labels = inputs.to(device), cls_labels.to(device), reg_labels.to(device)
                cls_output, reg_output = model(inputs)
                loss = criterion(cls_output, cls_labels, reg_output, reg_labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # 计算验证集上的分类和回归指标
        eval_metrics = evaluate_model(model, val_loader, device)
        accuracy = eval_metrics.get('accuracy', 0.0)
        precision = eval_metrics.get('precision', 0.0)
        recall = eval_metrics.get('recall', 0.0)
        f1 = eval_metrics.get('f1', 0.0)
        mse = eval_metrics.get('mse', 0.0)
        mae = eval_metrics.get('mae', 0.0)
        dtw = eval_metrics.get('dtw', 0.0)

        # 打印当前 epoch 的训练/验证损失和指标
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
              f"Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f} | "
              f"MSE={mse:.4f}, MAE={mae:.4f}, DTW={dtw:.4f}")

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 记录日志
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mse': mse,
            'mae': mae,
            'dtw': dtw
        })

        # 检查是否为最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping: validation loss has not improved for {patience} epochs.")
                break

    return best_model_path, history
