import torch
import torch.nn as nn


class MyModel1(nn.Module):
    def __init__(self, inplace):
        # tain_data的数据是13列
        # 这里的inplace是指Relu激活函数中的参数
        super().__init__()
        self.my_model1 = nn.Sequential(
            nn.Linear(13, 64),
            # nn.Dropout1d(p=0.5, inplace=False),
            nn.ReLU(inplace),
            nn.Linear(64, 32),
            # nn.Dropout1d(p=0.5, inplace=False),
            nn.ReLU(inplace),
            nn.Linear(32, 8),
            # nn.Dropout1d(p=0.5, inplace=False),
            nn.ReLU(inplace),
            nn.Linear(8, 1),
        )

    def forward(self, data):
        data = data.to(torch.float32)
        return self.my_model1(data)
