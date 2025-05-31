# from bearing_dataset import BearingDataset
# from torch.utils.data import DataLoader

# ds = BearingDataset(data_dir='.', split='train')
# loader = DataLoader(ds, batch_size=32, shuffle=False)

# # 取第一个 batch
# x_batch, y_class_batch, y_trend_batch = next(iter(loader))

# print("x_batch.shape:",    x_batch.shape)    # 预期 [32, Lhist, 1]
# print("y_class.shape:",    y_class_batch.shape)  # 预期 [32]
# print("y_trend.shape:",    y_trend_batch.shape)  # 预期 [32, 1024]


#!/usr/bin/env python3
# test.py — 在 test 集上跑分类预测并打印 classification_report

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# 请根据你的项目结构调整导入路径
from models.model import MultiTaskModel
from bearing_dataset import BearingDataset

def parse_args():
    p = argparse.ArgumentParser("Test classification report for MultiTaskModel")
    p.add_argument("--data_dir",    type=str, default=".",     help=".joblib 文件所在目录")
    p.add_argument("--model_path",  type=str, default="logs/best_model.pth", help="最优模型权重路径")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",      type=str, default="cuda", help="'cuda' or 'cpu'")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 加载模型结构＋权重
    model = MultiTaskModel()  # 使用 __init__ 中的默认参数
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # 2) 构建 Test DataLoader
    test_ds = BearingDataset(data_dir=args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    # 3) 推理，仅分类分支
    with torch.no_grad():
        for x, y_class, _ in test_loader:
            x = x.to(device)               # [batch, seq_len, 1]
            logits, _ = model(x)           # returns (class_out, reg_out)
            preds = logits.argmax(dim=1)   # [batch]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_class.tolist())

    # 4) 打印分类报告
    print("===== Classification Report on TEST set =====")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    main()

