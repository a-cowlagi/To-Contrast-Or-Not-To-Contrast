#!/usr/bin/env python3
"""
Implementation of Wide-Resnets for Multi-task learning.
Adapted from an open-source implementation.
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch import Tensor

class BasicBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 stride: int,
                 drop_rate: float = 0.0, 
                 layer_norm: bool = True) -> None:
        nn.Module.__init__(self)
        if layer_norm:
            self.bn1 = nn.GroupNorm(1, in_planes)
            self.bn2 = nn.GroupNorm(1, out_planes)
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(out_planes)

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = nn.Conv2d(in_planes, out_planes,
                                      kernel_size=1, stride=stride,
                                      padding=0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self,
                 nb_layers: int,
                 in_planes: int,
                 out_planes: int,
                 block: BasicBlock,
                 stride: int,
                 drop_rate: float,
                 layer_norm: bool) -> None:
        nn.Module.__init__(self)
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, drop_rate, layer_norm)

    def _make_layer(self,
                    block: BasicBlock,
                    in_planes: int,
                    out_planes: int,
                    nb_layers: int,
                    stride: int,
                    drop_rate: float,
                    layer_norm: bool) -> nn.Sequential:
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, layer_norm))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class WideResnet(nn.Module):
    """
    Wide-Resnet (https://arxiv.org/abs/1605.07146) for multiple tasks.
    This implementation assumes all tasks have the same number of classes.
    """
    def __init__(self,
                 num_cls: int,
                 net_cfg,
                 cls_per_task: int = None) -> None:
        """
        Args:
            - num_cls: Number of classes
            - cls_per_task: Number of classes per task (only needed 
        """
        super(WideResnet, self).__init__()
        assert((net_cfg.depth - 4) % 6 == 0)

        inp_channels = 3
        nChannels = [16, 16*net_cfg.width, 32*net_cfg.width, 64*net_cfg.width]

        if hasattr(net_cfg, "euclid"):
            self.euclid = net_cfg.euclid
        else:
            self.euclid = False

        if hasattr(net_cfg, "fc_bias"):
            self.fc_bias = net_cfg.fc_bias
        else:
            self.fc_bias = True

        layer_norm = net_cfg.layer_norm
        n = (net_cfg.depth - 4) // 6
        block = BasicBlock

        self.cls_per_task = cls_per_task

        self.conv1 = nn.Conv2d(inp_channels, nChannels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block,
                                   1, net_cfg.dropout, layer_norm)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block,
                                   2, net_cfg.dropout, layer_norm)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block,
                                   2, net_cfg.dropout, layer_norm)

        # global average pooling to accomodate flexible input image sizes
        if layer_norm:
            self.bn1 = nn.GroupNorm(1, nChannels[3])
        else:
            self.bn1 = nn.BatchNorm2d(nChannels[3])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(nChannels[3], num_cls, bias=self.fc_bias)

        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if self.fc_bias:
                    m.bias.data.zero_()

    def forward(self,
                x: Tensor,
                task = None) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.pool(out)
        out = out.view(-1, self.nChannels)
        self.rep = out

        # If given tasks, reshape to multi-task
        if self.euclid:
            logits = self.euclid_dist(self.fc.weight, out)
        else:
            logits = self.fc(out)
        if task is not None:

            if isinstance(task, int):
                logits = logits[:, self.cls_per_task*task:self.cls_per_task*(task+1)]
            else:
                bsize = logits.size(0)
                logits = logits.reshape(bsize, -1, self.cls_per_task)
                logits = logits[torch.arange(bsize), list(task), :]

        return logits

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if name in ["fc.weight", "fc.bias"]:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def euclid_dist(self, proto, rep):
        n = rep.shape[0]
        k = proto.shape[0]
        rep = rep.unsqueeze(1).expand(n, k, -1)
        proto = proto.unsqueeze(0).expand(n, k, -1)
        logits = -((rep - proto)**2).sum(dim=2)
        return logits

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_blocks(self, num_blocks_freeze):

        for name, param in self.named_parameters():
            name = name.split(".")

            if name[0] == "conv1":
                param.requires_grad = False
            elif name[0] == "bn1" or name[0] == "fc":
                param.requires_grad = True
            else:
                block_num = int(name[0][-1])
                if block_num <= num_blocks_freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    def load_half_model(self, ckpt, num_blocks_load):
        net_state = self.state_dict()
        for name, param in ckpt.items():
            if name not in net_state:
                 continue

            name_elem = name.split(".")
            if name[0] == "conv1":
                continue
            elif name_elem[0] != "bn1" and name_elem[0] != "fc":
                block_num = int(name_elem[0][-1])
                if block_num > num_blocks_load:
                    continue

            net_state[name].copy_(param)
