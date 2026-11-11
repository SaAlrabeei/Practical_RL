"""MultiTaskModel: shared EfficientNet-B4 encoder + U-Net decoder + per-tooth classification head."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskModel(nn.Module):
    """
    * Segmentation branch: U-Net decoder on the shared encoder -> 1-channel logit map
      (gingival-recession region).
    * Classification branch: each tooth-level ROI crop is passed through the *same*
      encoder; the deepest feature map goes through
      AdaptiveAvgPool -> BatchNorm -> Dropout -> FC(256) -> ReLU -> Dropout -> FC(3).
    """

    def __init__(self, encoder="efficientnet-b4", encoder_weights="imagenet",
                 num_classes=3, cls_hidden=256, dropout=0.3):
        super().__init__()
        self.unet = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                             in_channels=3, classes=1, activation=None)
        feat_ch = self.unet.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(feat_ch),
            nn.Dropout(dropout),
            nn.Linear(feat_ch, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    @property
    def encoder(self):
        return self.unet.encoder

    def segment(self, images: torch.Tensor) -> torch.Tensor:
        return self.unet(images)

    def classify(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.shape[0] == 0:
            return crops.new_zeros((0, self.cls_head[-1].out_features))
        feats = self.encoder(crops)[-1]
        if self.training and crops.shape[0] == 1:
            # BatchNorm1d cannot use batch statistics with a single sample
            self.cls_head[2].eval()
            out = self.cls_head(feats)
            self.cls_head[2].train()
            return out
        return self.cls_head(feats)

    def forward(self, images, crops):
        return self.segment(images), self.classify(crops)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * target).sum(dims)
    union = p.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + eps) / (union + eps)).mean()


class MultiTaskLoss(nn.Module):
    """total = (0.5 * Dice + 0.5 * BCE) + 0.8 * class-weighted CE (manuscript settings)."""

    def __init__(self, class_weights=None, dice_w=0.5, bce_w=0.5, cls_w=0.8):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.register_buffer("cw", class_weights if class_weights is not None else torch.ones(3))
        self.dice_w, self.bce_w, self.cls_w = dice_w, bce_w, cls_w

    def forward(self, seg_logits, masks, cls_logits, labels):
        seg = self.dice_w * dice_loss(seg_logits, masks) + self.bce_w * self.bce(seg_logits, masks)
        if labels.numel():
            cls = F.cross_entropy(cls_logits, labels, weight=self.cw)
        else:
            cls = seg_logits.new_zeros(())
        return seg + self.cls_w * cls, seg.detach(), cls.detach()
