# bearing_dataset.py

import os
import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# === 第一部分：从 MAT 文件生成 CSV，并划分保存为 .joblib ===

# 目录 & 文件名 设置
BASE_DIR = 'dataset/matfiles'
MAT_FILES = [
    '0_0.mat', '7_1.mat', '7_2.mat', '7_3.mat',
    '14_1.mat', '14_2.mat', '14_3.mat',
    '21_1.mat', '21_2.mat', '21_3.mat'
]
MAT_FIELDS = [
    'X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time',
    'X169_DE_time', 'X185_DE_time', 'X197_DE_time', 'X209_DE_time',
    'X222_DE_time', 'X234_DE_time'
]
COLUMN_NAMES = [
    'de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer',
    'de_14_inner', 'de_14_ball', 'de_14_outer',
    'de_21_inner', 'de_21_ball', 'de_21_outer'
]
# 标签映射
# —— 10 类一一映射 —— 
LABEL_MAP = {
    'de_normal'    : 0,
    'de_7_inner'   : 1,
    'de_7_ball'    : 2,
    'de_7_outer'   : 3,
    'de_14_inner'  : 4,
    'de_14_ball'   : 5,
    'de_14_outer'  : 6,
    'de_21_inner'  : 7,
    'de_21_ball'   : 8,
    'de_21_outer'  : 9,
}

def bandpass_filter(data, lowcut=200, highcut=5900, fs=12000, order=4):
    """带通滤波"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def normalize(data):
    """0–1 归一化"""
    mn, mx = data.min(), data.max()
    return (data - mn) / (mx - mn + 1e-8)

def create_dualtask_samples(signal, label, Lhist=1024, Lpred=1024, stride=512):
    """
    从一条归一化后的序列生成多任务样本：
      x: Lhist 点历史； y: Lpred 点未来； 最后附带 label
    返回 samples[N, Lhist+Lpred+1] 和 condition[N]
    """
    L = Lhist + Lpred
    samples = []
    for i in range(0, len(signal) - L + 1, stride):
        x = signal[i:i+Lhist]
        y = signal[i+Lhist:i+L]
        samples.append(np.concatenate([x, y, [label]]))
    return np.stack(samples, axis=0)

def process_dataset(csv_file='data_12k_10c.csv',
                    Lhist=1024, Lpred=1024, stride=512,
                    split_rate=(0.6, 0.2, 0.2)):
    """
    1) 读取 CSV
    2) 对每一列按 LABEL_MAP 处理、滤波、切片构造样本
    3) 分层抽样划分 train/val/test
    4) 保存归一化后张量到 .joblib： X、Yclass、Ytrend
    """

    df = pd.read_csv(csv_file)
    all_samples, all_labels = [], []
    
    for col in df.columns:
        if col not in LABEL_MAP: continue
        sig = df[col].values
        sig = normalize(bandpass_filter(sig))
        lab = LABEL_MAP[col]
        samp = create_dualtask_samples(sig, lab, Lhist, Lpred, stride)
        all_samples.append(samp)
        all_labels += [lab] * len(samp)

    data = np.vstack(all_samples)  # [总样本, Lhist+Lpred+1]
    # 最后一列是 label
    X = data[:, :Lhist]
    Ytrend = data[:, Lhist:Lhist+Lpred]
    Yclass = data[:, -1].astype(np.int64)

    # 分层抽样 split
    idx = np.arange(len(X))
    train_idx, tmp = train_test_split(idx, test_size=split_rate[1]+split_rate[2],
                                       stratify=Yclass, random_state=42)
    val_idx, test_idx = train_test_split(tmp, test_size=split_rate[2]/(split_rate[1]+split_rate[2]),
                                         stratify=Yclass[tmp], random_state=42)

    splits = {
        'train': train_idx,
        'val':   val_idx,
        'test':  test_idx
    }

    # 保存
    for split, ids in splits.items():
        dump(X[ids],       f'{split}X_dualtask.joblib')
        dump(Yclass[ids],  f'{split}Yclass_dualtask.joblib')
        dump(Ytrend[ids],  f'{split}Ytrend_dualtask.joblib')
        print(f"✅ {split} 集: 样本数 {len(ids)}, 文件 {split}X/Yclass/Ytrend_dualtask.joblib")

    # 打印总样本统计    
    total_samples = len(X)
    print(f"\n📊 数据集总样本数: {total_samples}")
    print(f"➡️ 训练集: {len(train_idx)} 条")
    print(f"➡️ 验证集: {len(val_idx)} 条")
    print(f"➡️ 测试集: {len(test_idx)} 条")

if __name__ == "__main__":
    # 先生成 CSV（如果已存在，可注释掉这一块）
    # -------------------------------------------------------
    # data_12k_10c = pd.DataFrame()
    # for i, fn in enumerate(MAT_FILES):
    #     mat = loadmat(os.path.join(BASE_DIR, fn))
    #     arr = mat[MAT_FIELDS[i]].reshape(-1)[:119808]
    #     data_12k_10c[COLUMN_NAMES[i]] = arr
    # data_12k_10c.to_csv('data_12k_10c.csv', index=False)
    # print("✅ data_12k_10c.csv 已生成")
    # -------------------------------------------------------
    # 划分并保存 .joblib
    process_dataset()

# === 第二部分：PyTorch Dataset 封装 ===

class BearingDataset(Dataset):
    """
    从上面保存的 .joblib 文件中加载 dual-task 数据。
    返回：(X_tensor [1,Lhist], y_class_tensor, y_trend_tensor [Lpred])
    """

    def __init__(self, data_dir='.', split='train'):
        assert split in ('train','val','test')
        fX, fC, fT = (
            f'{split}X_dualtask.joblib',
            f'{split}Yclass_dualtask.joblib',
            f'{split}Ytrend_dualtask.joblib'
        )
        self.X  = torch.from_numpy(load(os.path.join(data_dir, fX))).float()    # [N, Lhist]
        self.yc = torch.from_numpy(load(os.path.join(data_dir, fC))).long()     # [N]
        self.yt = torch.from_numpy(load(os.path.join(data_dir, fT))).float()    # [N, Lpred]

        # 如果你的模型期望 [batch,1,seq_len]，可以在 __getitem__ 里 unsqueeze
        assert len(self.X)==len(self.yc)==len(self.yt)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        # 在最后一维增加通道：输出 (Lhist, 1)，DataLoader 批量后即 (batch, Lhist, 1)
        x = self.X[idx].unsqueeze(-1)
        return x, self.yc[idx], self.yt[idx]
