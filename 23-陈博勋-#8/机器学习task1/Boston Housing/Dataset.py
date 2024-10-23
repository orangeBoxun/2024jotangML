import torch
from torch.utils.data import Dataset

# 定义可以直接用于dataloader的数据集（实现__len__和__getitem__方法）
# 我们自定义的dataset中，data和label是分开存储的
class MyDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label
