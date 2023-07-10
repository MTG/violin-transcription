import numpy as np
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, f, w, s, d, in_channels):
        super().__init__()
        p1 = d*(w - 1) // 2
        p2 = d*(w - 1) - p1
        self.pad = nn.ZeroPad2d((0, 0, p1, p2))

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=(s, 1), dilation=(d, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class NoPadConvBlock(nn.Module):
    def __init__(self, f, w, s, d, in_channels):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=(s, 1),
                                dilation=(d, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class TinyPathway(nn.Module):
    def __init__(self, dilation=1, hop=256, localize=False,
                 model_capacity="full", n_layers=6, chunk_size=256):
        super().__init__()

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]
        self.layers = [1, 2, 3, 4, 5, 6]
        self.layers = self.layers[:n_layers]
        filters = [n * capacity_multiplier for n in [32, 8, 8, 8, 8, 8]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 32, 32]
        strides = self.deter_dilations(hop//(4*(2**n_layers)), localize=localize)
        strides[0] = strides[0]*4  # apply 4 times more stride at the first layer
        dilations = self.deter_dilations(dilation)

        for i in range(len(self.layers)):
            f, w, s, d, in_channel = filters[i + 1], widths[i], strides[i], dilations[i], filters[i]
            self.add_module("conv%d" % i, NoPadConvBlock(f, w, s, d, in_channel))
        self.chunk_size = chunk_size
        self.input_window, self.hop = self.find_input_size_for_pathway()
        self.out_dim = filters[n_layers]

    def find_input_size_for_pathway(self):
        def find_input_size(output_size, kernel_size, stride, dilation, padding):
            num = (stride*(output_size-1)) + 1
            input_size = num - 2*padding + dilation*(kernel_size-1)
            return input_size
        conv_calc, n = {}, 0
        for i in self.layers:
            layer = self.__getattr__("conv%d" % (i-1))
            for mm in layer.modules():
                if hasattr(mm, 'kernel_size'):
                    try:
                        d = mm.dilation[0]
                    except TypeError:
                        d = mm.dilation
                    conv_calc[n] = [mm.kernel_size[0], mm.stride[0], 0, d]
                    n += 1
        out = self.chunk_size
        hop = 1
        for n in sorted(conv_calc.keys())[::-1]:
            kernel_size_n, stride_n, padding_n, dilation_n = conv_calc[n]
            out = find_input_size(out, kernel_size_n, stride_n, dilation_n, padding_n)
            hop = hop*stride_n
        return out, hop

    def deter_dilations(self, total_dilation, localize=False):
        n_layers = len(self.layers)
        if localize:  # e.g., 32*1023 window and 3 layers -> [1, 1, 32]
            a = [total_dilation] + [1 for _ in range(n_layers-1)]
        else:  # e.g., 32*1023 window and 3 layers -> [4, 4, 2]
            total_dilation = int(np.log2(total_dilation))
            a = []
            for layer in range(n_layers):
                this_dilation = int(np.ceil(total_dilation/(n_layers-layer)))
                a.append(2**this_dilation)
                total_dilation = total_dilation - this_dilation
        return a[::-1]

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)
        x = x.permute(0, 3, 2, 1)
        return x
