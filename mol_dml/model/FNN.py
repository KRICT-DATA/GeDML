import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(dim_in, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, dim_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)

        return h
