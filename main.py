# main.py

import argparse
import os
import torch
from torch.utils.data import DataLoader

# 你的模块路径，根据项目实际路径导入
from bearing_dataset import BearingDataset
from models.multitask_model import MultiTaskModel
from training.train import train
from training.evaluate import evaluate_model
import logging

logging.basicConfig(filename='training.log', level=logging.INFO)

def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate multi-task Transformer on CWRU data")
    # 数据和模型配置
    p.add_argument("--data_dir",     type=str, default="dataset/Data", help="Path to .npz files")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--patience",     type=int, default=10)
    p.add_argument("--sched_patience", type=int, default=5)
    p.add_argument("--out_dir",      type=str, default="outputs", help="Where to save models and logs")
    # 模型超参
    p.add_argument("--embed_dim",    type=int, default=128)
    p.add_argument("--heads",        type=int, default=8)
    p.add_argument("--layers",       type=int, default=4)
    p.add_argument("--kernel_sizes", type=int, nargs="+", default=[3,5,9])
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 数据加载
    # 构造文件列表（此处只示例 1797 RPM 四类文件，按需扩展）
    files = [
        os.path.join(args.data_dir, "1797_Normal.npz"),
        os.path.join(args.data_dir, "1797_IR_21_DE12.npz"),
        os.path.join(args.data_dir, "1797_B_14_DE12.npz"),
        os.path.join(args.data_dir, "1797_OR@12_21_DE12.npz"),
    ]
    # Dataset 返回 (X_hist, Y_pred, y_cls)
    dataset = BearingDataset(file_names=files,
                             Lhist=1024, Lpred=150, step=512)
    # 划分训练/验证
    total = len(dataset)
    n_train = int(0.8 * total)
    n_val   = total - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 2. 模型构建
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(
        in_channels=1,
        embed_dim=args.embed_dim,
        kernel_sizes=tuple(args.kernel_sizes),
        num_layers=args.layers,
        num_heads=args.heads
    )

    # 3. 训练
    best_model_path, history = train(
        train_loader, val_loader, model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        scheduler_patience=args.sched_patience,
        weight_decay=0
    )
    print(f"Best model saved to: {best_model_path}")

    # 4. 最终评估
    # 重新加载最佳权重
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    eval_results = evaluate_model(model, val_loader, device, log_dir=args.out_dir)
    print("Final evaluation on validation set:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")

    # 5. 可选：保存训练历史到文件
    import json
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
