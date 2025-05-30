from bearing_dataset import BearingDataset
from torch.utils.data import DataLoader

ds = BearingDataset(data_dir='.', split='train')
loader = DataLoader(ds, batch_size=32, shuffle=False)

# 取第一个 batch
x_batch, y_class_batch, y_trend_batch = next(iter(loader))

print("x_batch.shape:",    x_batch.shape)    # 预期 [32, Lhist, 1]
print("y_class.shape:",    y_class_batch.shape)  # 预期 [32]
print("y_trend.shape:",    y_trend_batch.shape)  # 预期 [32, 1024]
