import torch
from torch import nn
import math


class CARHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_tower = self.cls_tower(x)
        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)


        bbox_tower = self.bbox_tower(x)
        # centerness = self.centerness(bbox_tower)
        bbox_reg = self.bbox_pred(bbox_tower)
        bbox_reg = torch.exp(bbox_reg)

        return logits, bbox_reg, centerness


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale






# import torch
# from torch import nn
# import torch.nn.functional as F
# import math
#
#
# class CARHead(torch.nn.Module):
#     def __init__(self, cfg, in_channels):
#         """
#         Arguments:
#             in_channels (int): number of channels of the input feature
#         """
#         super(CARHead, self).__init__()
#         # TODO: Implement the sigmoid version first.
#         num_classes = cfg.TRAIN.NUM_CLASSES
#         self.weighted = True
#         # 每一层的权重
#         if self.weighted:
#             self.cls_weight = nn.Parameter(torch.ones(3))
#             self.loc_weight = nn.Parameter(torch.ones(3))
#             self.cen_weight = nn.Parameter(torch.ones(3))
#         self.loc_scale = nn.Parameter(torch.ones(3))
#
#         cls_tower_2 = []
#         bbox_tower_2 = []
#
#         cls_tower_3 = []
#         bbox_tower_3 = []
#
#         cls_tower_4 = []
#         bbox_tower_4 = []
#         for i in range(cfg.TRAIN.NUM_CONVS):
#             cls_tower_2.append(nn.Conv2d( in_channels, in_channels, kernel_size=3, stride=1, padding=1))
#             cls_tower_2.append(nn.GroupNorm(32, in_channels))
#             cls_tower_2.append(nn.ReLU())
#             bbox_tower_2.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
#             bbox_tower_2.append(nn.GroupNorm(32, in_channels))
#             bbox_tower_2.append(nn.ReLU())
#
#             cls_tower_3.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
#             cls_tower_3.append(nn.GroupNorm(32, in_channels))
#             cls_tower_3.append(nn.ReLU())
#             bbox_tower_3.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
#             bbox_tower_3.append(nn.GroupNorm(32, in_channels))
#             bbox_tower_3.append(nn.ReLU())
#
#             cls_tower_4.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
#             cls_tower_4.append(nn.GroupNorm(32, in_channels))
#             cls_tower_4.append(nn.ReLU())
#             bbox_tower_4.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
#             bbox_tower_4.append(nn.GroupNorm(32, in_channels))
#             bbox_tower_4.append(nn.ReLU())
#
#         self.add_module('cls_tower_2', nn.Sequential(*cls_tower_2))
#         self.add_module('bbox_tower_2', nn.Sequential(*bbox_tower_2))
#
#         self.add_module('cls_tower_3', nn.Sequential(*cls_tower_3))
#         self.add_module('bbox_tower_3', nn.Sequential(*bbox_tower_3))
#
#         self.add_module('cls_tower_4', nn.Sequential(*cls_tower_4))
#         self.add_module('bbox_tower_4', nn.Sequential(*bbox_tower_4))
#
#         self.cls_logits_2 = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
#         self.bbox_pred_2 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
#         self.centerness_2 = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
#
#         self.cls_logits_3 = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
#         self.bbox_pred_3 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
#         self.centerness_3 = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
#
#         self.cls_logits_4 = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
#         self.bbox_pred_4 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
#         self.centerness_4 = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
#
#         # initialization
#         for modules in [self.cls_tower_2, self.bbox_tower_2, self.cls_logits_2, self.bbox_pred_2, self.centerness_2,
#                         self.cls_tower_3, self.bbox_tower_3, self.cls_logits_3, self.bbox_pred_3, self.centerness_3,
#                         self.cls_tower_4, self.bbox_tower_4, self.cls_logits_4, self.bbox_pred_4, self.centerness_4]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#                     torch.nn.init.constant_(l.bias, 0)
#
#         # initialize the bias for focal loss
#         prior_prob = cfg.TRAIN.PRIOR_PROB
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         torch.nn.init.constant_(self.cls_logits_2.bias, bias_value)
#         torch.nn.init.constant_(self.cls_logits_3.bias, bias_value)
#         torch.nn.init.constant_(self.cls_logits_4.bias, bias_value)
#
#     def weighted_avg(self, lst, weight):
#         s = 0
#         for i in range(len(weight)):
#             s += lst[i] * weight[i]
#         return s
#
#     def forward(self, features):
#
#         cls = []
#         cen = []
#         loc = []
#         cls_tower_2 = self.cls_tower_2(features[0])
#         logits_2 = self.cls_logits_2(cls_tower_2)
#         centerness_2 = self.centerness_2(cls_tower_2)
#         bbox_reg_2 = torch.exp(self.bbox_pred_2(self.bbox_tower_2(features[0])) * self.loc_scale[0])
#         # bbox_reg_2 = torch.exp(self.bbox_pred_2(self.bbox_tower_2(features[0])))
#         cls.append(logits_2)
#         loc.append(bbox_reg_2)
#         cen.append(centerness_2)
#
#         cls_tower_3 = self.cls_tower_3(features[1])
#         logits_3 = self.cls_logits_3(cls_tower_3)
#         centerness_3 = self.centerness_3(cls_tower_3)
#         bbox_reg_3 = torch.exp(self.bbox_pred_3(self.bbox_tower_3(features[1])) * self.loc_scale[1])
#         # bbox_reg_3 = torch.exp(self.bbox_pred_3(self.bbox_tower_3(features[1])))
#         cls.append(logits_3)
#         loc.append(bbox_reg_3)
#         cen.append(centerness_3)
#
#         cls_tower_4 = self.cls_tower_4(features[2])
#         logits_4 = self.cls_logits_4(cls_tower_4)
#         centerness_4 = self.centerness_4(cls_tower_4)
#         bbox_reg_4 = torch.exp(self.bbox_pred_4(self.bbox_tower_4(features[2])) * self.loc_scale[2])
#         # bbox_reg_4 = torch.exp(self.bbox_pred_4(self.bbox_tower_4(features[2])))
#         cls.append(logits_4)
#         loc.append(bbox_reg_4)
#         cen.append(centerness_4)
#
#         # print(self.cls_weight)
#
#         if self.weighted:
#             cls_weight = F.softmax(self.cls_weight, 0)
#             loc_weight = F.softmax(self.loc_weight, 0)
#             cen_weight = F.softmax(self.cen_weight, 0)
#
#         cls = self.weighted_avg(cls, cls_weight)
#         loc = self.weighted_avg(loc, loc_weight)
#         cen = self.weighted_avg(cen, cen_weight)
#
#         return cls, loc, cen
#
#
# class Scale(nn.Module):
#     def __init__(self, init_value=1.0):
#         super(Scale, self).__init__()
#         self.scale = nn.Parameter(torch.FloatTensor([init_value]))
#
#     def forward(self, input):
#         return input * self.scale
