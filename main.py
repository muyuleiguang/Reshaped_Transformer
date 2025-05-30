# main.py
import os
import torch
import torch.optim as optim 
import argparse
import logging
from torch.utils.data import DataLoader
from models.model import MultiTaskModel
from training.train import train
from training.evaluate import evaluate
from bearing_dataset import BearingDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Bearing Fault Diagnosis")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="运行模式：'train' 执行训练流程；'eval' 仅加载最优模型并评估")
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--data_dir', type=str, default='.', help='.joblib 数据文件所在目录')
    parser.add_argument('--log_dir', type=str, default='logs/', help='日志及模型存储目录')
    args = parser.parse_args()

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'main.log'), mode='w'),
            logging.StreamHandler()
        ]
    )

    # 准备 DataLoader：
    train_loader = DataLoader(
        BearingDataset(data_dir=args.data_dir, split='train'),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        BearingDataset(data_dir=args.data_dir, split='val'),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        BearingDataset(data_dir=args.data_dir, split='test'),
        batch_size=args.batch_size, shuffle=False
    )
    

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    model = MultiTaskModel().to(device)

    if args.mode == 'train':
        # ----------------- 训练 + 测试评估 -----------------
        logging.info("开始训练 …")
        best_model = train(
            model, train_loader, val_loader,
            optimizer=optim.Adam(model.parameters(), lr=args.learning_rate),
            # TensorBoard 日志目录
            log_dir=args.log_dir,
            epochs=args.epochs,
            patience=20
        )
        logging.info("训练完毕，开始测试集评估 …")
        test_metrics = evaluate(best_model, test_loader)
        logging.info(f"Test结果：{test_metrics}")
        # 最优模型已在 train() 中存储于 logs/best_model.pth
    else:
        # ----------------- 仅评估 -----------------
        ckpt = os.path.join(args.log_dir, "best_model.pth")
        logging.info(f"加载最优模型权重：{ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        logging.info("开始在测试集上评估 …")
        test_metrics = evaluate(model, test_loader)
        logging.info(f"Test结果：{test_metrics}")

    logging.info("流程结束。")
