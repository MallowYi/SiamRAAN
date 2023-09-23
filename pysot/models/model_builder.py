# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from pysot.models.attention import Cross_attention
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
                                     
        self.cross_P2 = Cross_attention.ATTBlock(512)
        self.cross_P3 = Cross_attention.ATTBlock(1024)
        self.cross_P4 = Cross_attention.ATTBlock(2048)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)


    def template(self, z):
        self.zf_ = self.backbone(z)

        (self.zf_1, self.zf_w) = self.attention_model_eca(self.zf_)


    def track(self, x):
        xf_ = self.backbone(x)

        (xf_1, xf_w) = self.attention_model_eca(xf_)
        zf_2 = self.attention_model_mix(self.zf_, xf_w)
        xf_2 = self.attention_model_mix(xf_, self.zf_w)

        zf0 = self.zf_[0] + self.zf_1[0] + zf_2[0]
        zf1 = self.zf_[1] + self.zf_1[1] + zf_2[1]
        zf2 = self.zf_[2] + self.zf_1[2] + zf_2[2]
        zf = []
        zf.append(zf0)
        zf.append(zf1)
        zf.append(zf2)

        xf0 = xf_[0] + xf_1[0] + xf_2[0]
        xf1 = xf_[1] + xf_1[1] + xf_2[1]
        xf2 = xf_[2] + xf_1[2] + xf_2[2]
        xf = []
        xf.append(xf0)
        xf.append(xf1)
        xf.append(xf2)


        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
            zf = self.neck(zf)

        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        return {
            'cls': cls,
            'loc': loc,
            'cen': cen
        }
        
    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def attention_model_eca(self, features):
        out = []
        out_weight = []
        for i in range(len(features)):
            if i == 0:
                w = self.cross_P2(features[i])
                feature = w * features[i] + features[i]
            elif i == 1:
                w = self.cross_P3(features[i])
                feature = w * features[i] + features[i]
            else:
                w = self.cross_P4(features[i])
                feature = w * features[i] + features[i]
            out_weight.append(w)
            out.append(feature)
        return out, out_weight

    def attention_model_mix(self, features, w):
        out = []
        for i in range(len(features)):
            if i == 0:
                feature = w[i] * features[i] + features[i]
            elif i == 1:
                feature = w[i] * features[i] + features[i]
            else:
                feature = w[i] * features[i] + features[i]
            out.append(feature)
        return out

    def forward(self, data):
        """
         only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf_ = self.backbone(template) #[512,15,15] [1024,15,15] [2048,15,15]
        xf_ = self.backbone(search) #[512,31,31] [1024,31,31] [2048,31,31]
        (zf_1, zf_w) = self.attention_model_eca(zf_)
        (xf_1, xf_w) = self.attention_model_eca(xf_)
        zf_2 = self.attention_model_mix(zf_, xf_w)
        xf_2 = self.attention_model_mix(xf_, zf_w)

        zf0 = zf_[0] + zf_1[0] + zf_2[0]
        zf1 = zf_[1] + zf_1[1] + zf_2[1]
        zf2 = zf_[2] + zf_1[2] + zf_2[2]

        zf = []
        zf.append(zf0)
        zf.append(zf1)
        zf.append(zf2)

        xf0 = xf_[0] + xf_1[0] + xf_2[0]
        xf1 = xf_[1] + xf_1[1] + xf_2[1]
        xf2 = xf_[2] + xf_1[2] + xf_2[2]

        xf = []
        xf.append(xf0)
        xf.append(xf1)
        xf.append(xf2)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf) #[512,15,15] [1024,15,15] [2048,15,15] -> [256,7,7] [256,7,7] [256,7,7]
            xf = self.neck(xf) #[512,31,31] [1024,31,31] [2048,31,31] -> [256,31,31] [256,31,31] [256,31,31]

        features = self.xcorr_depthwise(xf[0],zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)


        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs