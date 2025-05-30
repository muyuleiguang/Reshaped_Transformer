# train.py
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from training.evaluate import evaluate
from training.loss import compute_loss

def train(model, train_loader, val_loader, optimizer, log_dir='logs/', epochs=100, patience=10):
    """
    模型训练函数，包含训练循环、验证评估、早停机制和TensorBoard记录。
    参数：
        model: 多任务模型实例
        train_loader: 训练集的DataLoader
        val_loader: 验证集的DataLoader
        optimizer: 优化器（如Adam）
        log_dir: TensorBoard日志输出目录
        epochs: 最大训练轮数
        patience: 验证集无改进的早停轮数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    writer = SummaryWriter(log_dir=log_dir)  # 初始化TensorBoard记录器
    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for X, y_class, y_reg in train_loader:
            X = X.to(device)
            y_class = y_class.to(device)
            y_reg = y_reg.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            # 计算损失
            loss, class_loss, reg_loss = compute_loss(outputs, (y_class, y_reg))
            loss.backward()
            optimizer.step()

            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_train_loss = running_loss / total_samples

        # 在验证集上评估
        metrics = evaluate(model, val_loader)
        val_total_loss = metrics['loss_total']
        val_class_loss = metrics['loss_class']
        val_reg_loss   = metrics['loss_reg']
        val_acc        = metrics['accuracy']
        val_prec      = metrics['precision']
        val_rec       = metrics['recall']
        val_f1        = metrics['f1_score']
        val_mse        = metrics['mse']
        val_mae        = metrics['mae']
        val_dtw       = metrics['dtw']
        
        # 打印日志
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={val_total_loss:.4f} "
              f"(Class={val_class_loss:.4f}, Reg={val_reg_loss:.4f}), "
              f"Acc={val_acc*100:.2f}%, "
              f"Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}, "
              f"MSE={val_mse:.4f}, MAE={val_mae:.4f}, DTW={val_dtw:.4f}")

        # 写入TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Total", val_total_loss, epoch)
        writer.add_scalar("Loss/Val_Classification", val_class_loss, epoch)
        writer.add_scalar("Loss/Val_Regression", val_reg_loss, epoch)
        writer.add_scalar("Metrics/Val_Accuracy", val_acc, epoch)
        writer.add_scalar("Metrics/Val_MSE", val_mse, epoch)
        writer.add_scalar("Metrics/Val_MAE", val_mae, epoch)
        writer.add_scalar("Metrics/Val_Precision",   val_prec, epoch)
        writer.add_scalar("Metrics/Val_Recall",      val_rec, epoch)
        writer.add_scalar("Metrics/Val_F1",          val_f1, epoch)
        writer.add_scalar("Metrics/Val_DTW",         val_dtw, epoch)

        # 检查是否为当前最佳
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_model_wts = {name: param.clone() for name, param in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))  # 保存最优模型
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"验证集{patience}个epoch无改进，提前停止训练。")
                break

    writer.close()

    if best_model_wts:
        model.load_state_dict(best_model_wts)
        print("已加载验证集上表现最好的模型权重")
    return model