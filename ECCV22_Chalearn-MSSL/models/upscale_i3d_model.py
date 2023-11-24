import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["InceptionI3d"]


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,ms.shaw
        # out_t = np.ceil(float(t) / float(self.stride[0]))
        # out_h = np.ceil(float(h) / float(self.stride[1]))
        # out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
        num_domains=1,
    ):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._num_domains = num_domains
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            if self._num_domains == 1:
                self.bn = nn.BatchNorm3d(
                    self._output_channels, eps=0.001, momentum=0.01
                )

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        # out_t = np.ceil(float(t) / float(self._stride[0]))
        # out_h = np.ceil(float(h) / float(self._stride[1]))
        # out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name, num_domains=1, activation_fn=F.relu):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
            activation_fn=activation_fn
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
            activation_fn=activation_fn
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
            activation_fn=activation_fn
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
            activation_fn=activation_fn
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
            activation_fn=activation_fn
        )
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
            activation_fn=activation_fn
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class UpscaleInceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=400,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=64,
        include_embds=False,
        ckpt_path=None,
        inc_logits=False,
        activation='relu'
    ):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatiotemporal_squeeze: Whether to squeeze the 2 spatial and 1 temporal dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          in_channels: Number of input channels (default 3 for RGB).
          dropout_keep_prob: Dropout probability (default 0.5).
          name: A string (optional). The name of this module.
          num_in_frames: Number of input frames (default 64).
          include_embds: Whether to return embeddings (default False).
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % final_endpoint)

        super().__init__()

        if activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'swish':
            act = torch.nn.SiLU()

        self._num_classes = num_classes
        self._spatiotemporal_squeeze = spatiotemporal_squeeze
        self._final_endpoint = final_endpoint
        self.include_embds = include_embds
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % self._final_endpoint)

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(
            192,
            [64, 96, 128, 16, 32, 32],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            256,
            [128, 128, 192, 32, 96, 64],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64,
            [192, 96, 208, 16, 48, 64],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64,
            [160, 112, 224, 24, 64, 64],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64,
            [128, 128, 256, 24, 64, 64],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64,
            [112, 144, 288, 32, 64, 64],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64,
            [256, 160, 320, 32, 128, 128],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128,
            [256, 160, 320, 32, 128, 128],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128,
            [384, 192, 384, 48, 128, 128],
            name + end_point,
            activation_fn=act,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Logits"

        last_duration = int(math.ceil(num_in_frames / 8))  # 8
        last_size = 7  # int(math.ceil(sample_width / 32))  # this is for 224
        # self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(dropout_keep_prob)

        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        self.build()

        if ckpt_path:
            ckpt = torch.load(ckpt_path)
            dict_ckpt = {}
            for key, value in ckpt["state_dict"].items():
                if "logits" not in key:
                    # print(key.replace("module.",""))
                    dict_ckpt[key.replace("module.", "")] = value
                elif inc_logits and "logits" in key:
                    dict_ckpt[key.replace("module.", "")] = value
                else:
                    print(key)
            res = self.load_state_dict(dict_ckpt, strict=False)
            print(res)

        self.spatial_downscaling_Mixed_4f = BasicBlock(
            inplanes=832,
            planes=832 // 2,
            stride=(2, 2),
            downsample=downsample_avg(
                in_channels=832, out_channels=832 // 2, kernel_size=3, stride=2
            ),
            act_layer= torch.nn.ReLU if act =='relu' else torch.nn.SiLU
        )

        self.spatial_downscaling_Mixed_3c = BasicBlock(
            inplanes=480,
            planes=480 // 2,
            stride=(4, 4),
            downsample=downsample_avg(
                in_channels=480, out_channels=480 // 2, kernel_size=3, stride=4
            ),
            act_layer= torch.nn.ReLU if act =='relu' else torch.nn.SiLU
        )

        self.up_Mixed_5c = nn.ConvTranspose3d(
            1024, 1024 // 2, kernel_size=(2, 1, 1), stride=(2, 1, 1)
        )

        self.merge_5c4f = DoubleConv(1024 // 2 + 832 // 2, 832, activation=act)
        self.conv_8 = nn.Conv3d(832, 832, kernel_size=1)

        self.up_5c4f = nn.ConvTranspose3d(
            832, 832 // 2, kernel_size=(2, 1, 1), stride=(2, 1, 1)
        )

        self.merge_5c4f3c = DoubleConv(832 // 2 + 480 // 2, 480, activation=act)
        self.conv_16 = nn.Conv3d(480, 480, kernel_size=1)

        self.up_5c4f3c = nn.ConvTranspose3d(
            480, 480 // 2, kernel_size=(2, 1, 1), stride=(2, 1, 1)
        )
        self.conv_32 = nn.Conv3d(480 // 2, 480 // 2, kernel_size=1)

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        dict_ends = {}
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
                if end_point in ["Mixed_3c", "Mixed_4f", "Mixed_5c"]:
                    dict_ends[end_point] = x

        x4 = x

        Mixed_5c = F.adaptive_avg_pool3d(
            dict_ends["Mixed_5c"], (dict_ends["Mixed_5c"].shape[2], 1, 1)
        )
        Mixed_5c = self.up_Mixed_5c(Mixed_5c)

        b, c, t, h, w = dict_ends["Mixed_4f"].shape
        Mixed_4f = self.spatial_downscaling_Mixed_4f(
            dict_ends["Mixed_4f"].permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        )
        bc2, c2, h2, w2 = Mixed_4f.shape
        Mixed_4f = F.adaptive_avg_pool3d(
            Mixed_4f.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4), (t, 1, 1)
        )

        mixed_5c4f = torch.cat([Mixed_5c, Mixed_4f], dim=1)
        mixed_5c4f = self.merge_5c4f(mixed_5c4f)
        x8 = self.conv_8(mixed_5c4f)

        b, c, t, h, w = dict_ends["Mixed_3c"].shape
        Mixed_3c = self.spatial_downscaling_Mixed_3c(
            dict_ends["Mixed_3c"].permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        )
        bc2, c2, h2, w2 = Mixed_3c.shape
        Mixed_3c = F.adaptive_avg_pool3d(
            Mixed_3c.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4), (t, 1, 1)
        )

        mixed_5c4f = self.up_5c4f(mixed_5c4f)
        mixed_5c4f3c = torch.cat([mixed_5c4f, Mixed_3c], dim=1)
        mixed_5c4f3c = self.merge_5c4f3c(mixed_5c4f3c)
        x16 = self.conv_16(mixed_5c4f3c)

        x32 = self.conv_32(self.up_5c4f3c(mixed_5c4f3c))

        return {
            "x32": x32,
            "x16": x16,
            "x8": x8,
            "x4": x4,
        }

        # [batch x featuredim x 1 x 1 x 1]


#         embds = self.dropout(self.avgpool(x))

#         # [batch x classes x 1 x 1 x 1]
#         x = self.logits(embds)
#         if self._spatiotemporal_squeeze:
#             # [batch x classes]
#             logits = x.squeeze(3).squeeze(3).squeeze(2)

#         # logits [batch X classes]
#         if self.include_embds:
#             return {"logits": logits, "embds": embds}
#         else:
#             return {"logits": logits}
from timm.models.resnet import BasicBlock, downsample_avg


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation=torch.nn.ReLU()):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(mid_channels),
            activation,
            nn.Conv3d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.double_conv(x)


if __name__ == "__main__":

    model = UpscaleInceptionI3d(activation='swish')
    print(model)
    x = torch.randn(2, 3, 32, 224, 224)
    with torch.no_grad():
        y = model(x)
