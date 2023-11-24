import torch.nn as nn
import torch


class TimeHead(nn.Module):
    def __init__(
        self, input_dim, num_classes, dropout_time=0.5):
        super(TimeHead, self).__init__()

        self.time_lin = torch.nn.Linear(input_dim, num_classes+1)
        self.dropout_time = torch.nn.Dropout(dropout_time)


    def forward(self, x):
        if len(x.shape)!=3:
            y_pool2d = torch.nn.functional.adaptive_avg_pool2d(x,1).squeeze(-1).squeeze(-1).permute(0,2,1)
        else:
            y_pool2d = x
        y_pool2d_drop = self.dropout_time(y_pool2d)
        time_pred = self.time_lin(y_pool2d_drop)

        return {
            "pool2d": y_pool2d,
            "time_out": time_pred
        }