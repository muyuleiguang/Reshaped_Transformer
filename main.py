import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from bearing_dataset import BearingDataset
from models import MultiTaskModel
from training import train, evaluate_model

def parse_args():
    """
    解析命令行参数。
    :return: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='Train a multi-task Transformer model on bearing data.')
    parser.add_argument('--data_dir', type=str, default='dataset/Data/matfiles', help='数据目录')
    parser.add_argument('--out_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='学习率调度耐心值')
    parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减')
    return parser.parse_args()

def main():
    """
    主函数，执行数据加载、模型训练和评估。
    """
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载数据集
    files = [
        f'{args.data_dir}/0_0.mat',
        f'{args.data_dir}/7_1.mat',
        f'{args.data_dir}/7_2.mat',
        f'{args.data_dir}/7_3.mat',
        f'{args.data_dir}/14_1.mat',
        f'{args.data_dir}/14_2.mat',
        f'{args.data_dir}/14_3.mat',
        f'{args.data_dir}/21_1.mat',
        f'{args.data_dir}/21_2.mat',
        f'{args.data_dir}/21_3.mat',
    ]
    dataset = BearingDataset(files, Lhist=1024, Lpred=1024, step=512)
    X, Y_pred, y, file_indices = dataset.load_data()

    # 分层划分数据集
    train_idx, temp_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.4,  # 40% 用于验证+测试
        stratify=file_indices,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # 临时集的一半，即20%
        stratify=file_indices[temp_idx],
        random_state=42
    )

    # 创建子集
    train_X, train_Y_pred, train_y = X[train_idx], Y_pred[train_idx], y[train_idx]
    val_X, val_Y_pred, val_y = X[val_idx], Y_pred[val_idx], y[val_idx]
    test_X, test_Y_pred, test_y = X[test_idx], Y_pred[test_idx], y[test_idx]

    # 转换为 TensorDataset
    train_dataset = TensorDataset(
        torch.from_numpy(train_X).float().unsqueeze(1),
        torch.from_numpy(train_y).long(),
        torch.from_numpy(train_Y_pred).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_X).float().unsqueeze(1),
        torch.from_numpy(val_y).long(),
        torch.from_numpy(val_Y_pred).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_X).float().unsqueeze(1),
        torch.from_numpy(test_y).long(),
        torch.from_numpy(test_Y_pred).float()
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(
        in_channels=1,
        embed_dim=128,
        kernel_sizes=(3, 5, 9),
        num_layers=4,
        num_heads=8,
        dim_feedforward=256,
        local_window_size=5
    ).to(device)
    print("模型结构:", model)
    
