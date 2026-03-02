import torch.nn.functional as F 
from torch.nn import Linear
import torch

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64):
        super().__init__()
        self.lin1_src = Linear(in_channels, hidden)
        self.lin1_dst = Linear(in_channels, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.lin_final = Linear(hidden, out_channels)

    def forward(self, z_src, z_dst):
        h = self.lin1_src(z_src) + self.lin1_dst(z_dst).relu()
        h = self.lin2(h).relu()
        h = F.dropout(h, p=0.1, training=self.training)
        return self.lin_final(h)