import torch
from torch import nn


class NPCC(nn.Module):
    def __init__(self, positive=None):
        super(NPCC, self).__init__()
        self.positive = positive

    def forward(self, x, y, keep_batch = False):

        mean_x = x.mean(dim=[-1, -2], keepdim=True)
        mean_y = y.mean(dim=[-1, -2], keepdim=True)

        PCC = ((x-mean_x)*(y-mean_y)).sum(dim=[-1, -2])/(torch.sqrt(((x-mean_x)**2).sum(dim=[-1, -2]))*torch.sqrt(((y-mean_y)**2).sum(dim=[-1, -2])))

        if keep_batch:
            if self.positive is None:
                return -1*PCC.squeeze()
            else:
                return PCC.squeeze()
        else:

            if self.positive is None:
                return -1*torch.mean(PCC)
            else:
                return torch.mean(PCC)