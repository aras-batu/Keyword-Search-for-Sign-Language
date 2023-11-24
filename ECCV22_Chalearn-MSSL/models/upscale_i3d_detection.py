import torch.nn as nn
import torch

from models.upscale_i3d_model import UpscaleInceptionI3d
from models.time_head import TimeHead


class UpscaleI3dDectection(nn.Module):
    def __init__(self, i3d_model_params, head_model_params):
        super(UpscaleI3dDectection, self).__init__()

        self.backbone_model = UpscaleInceptionI3d(**i3d_model_params)

        self.head_4 = TimeHead(input_dim=1024, num_classes=60)
        self.head_8 = TimeHead(input_dim=832, num_classes=60)
        self.head_16 = TimeHead(input_dim=480, num_classes=60)
        self.head_32 = TimeHead(input_dim=240, num_classes=60)
        self.head_all = TimeHead(input_dim=1024 + 832 + 480 + 240, num_classes=60)

    def forward(self, x):
        y_r = self.backbone_model(x.permute(0, 2, 1, 3, 4))

        dict_res = {}

        dict_res["x4"] = self.head_4(y_r["x4"])
        dict_res["x8"] = self.head_8(y_r["x8"])
        dict_res["x16"] = self.head_16(y_r["x16"])
        dict_res["x32"] = self.head_32(y_r["x32"])

        x4 = nn.functional.interpolate(
            dict_res["x4"]["pool2d"].permute(0, 2, 1),
            size=32,
            scale_factor=None,
            mode="nearest",
        ).permute(0, 2, 1)
        x8 = nn.functional.interpolate(
            dict_res["x8"]["pool2d"].permute(0, 2, 1),
            size=32,
            scale_factor=None,
            mode="nearest",
        ).permute(0, 2, 1)
        x16 = nn.functional.interpolate(
            dict_res["x16"]["pool2d"].permute(0, 2, 1),
            size=32,
            scale_factor=None,
            mode="nearest",
        ).permute(0, 2, 1)
        x32 = nn.functional.interpolate(
            dict_res["x32"]["pool2d"].permute(0, 2, 1),
            size=32,
            scale_factor=None,
            mode="nearest",
        ).permute(0, 2, 1)

        x = torch.cat([x4, x8, x16, x32], dim=-1)
        dict_res["x"] = self.head_all(x)

        return dict_res
