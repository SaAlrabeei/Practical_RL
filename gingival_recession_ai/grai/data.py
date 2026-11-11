"""Dataset for multi-task gingival-recession segmentation + per-tooth Cairo classification.

Expected layout (created by ``cvat_to_dataset.py`` or ``make_synthetic_data.py``)::

    data/
      images/<image_id>.jpg|png        clinical intraoral photograph
      instances/<image_id>.png         uint16 instance mask: 0 = background,
                                       k = recession region of tooth k
      teeth.csv                        image_id, patient_id, instance_id, label (RT1/RT2/RT3)
      splits.csv                       image_id, split (train / val / test)

The binary segmentation target is ``instances > 0``; each instance gives one
tooth-level region of interest (ROI) and one RT label.
"""
from __future__ import annotations

import os
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ["RT1", "RT2", "RT3"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_image(root: str, image_id: str) -> str:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".bmp"):
        p = os.path.join(root, "images", image_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No image found for id {image_id}")


def resize_long_side(img: np.ndarray, size: int, interp) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def train_augmentations() -> A.Compose:
    """Augmentations listed in the manuscript (training set only)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.ElasticTransform(alpha=30, sigma=6, p=0.3),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.5),
        A.GaussNoise(p=0.3),
    ])


def normalize(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def box_from_mask(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def crop_roi(img: np.ndarray, box, pad: float, size: int, jitter: float = 0.0) -> np.ndarray:
    """Square-ish crop around ``box`` (x1, y1, x2, y2) with padding, resized to ``size``."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if jitter > 0:
        cx += random.uniform(-jitter, jitter) * bw
        cy += random.uniform(-jitter, jitter) * bh
        bw *= 1 + random.uniform(-jitter, jitter)
        bh *= 1 + random.uniform(-jitter, jitter)
    side = max(bw, bh) * (1 + 2 * pad)
    side = max(side, 16)
    nx1, ny1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    nx2, ny2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    # pad image if crop leaves the frame
    padl, padt = max(0, -nx1), max(0, -ny1)
    padr, padb = max(0, nx2 - w), max(0, ny2 - h)
    if padl or padt or padr or padb:
        img = cv2.copyMakeBorder(img, padt, padb, padl, padr, cv2.BORDER_CONSTANT, value=0)
        nx1, nx2, ny1, ny2 = nx1 + padl, nx2 + padl, ny1 + padt, ny2 + padt
    crop = img[ny1:ny2, nx1:nx2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


class GingivalRecessionDataset(Dataset):
    def __init__(self, root: str, split: str, work_size=768, seg_size=256, crop_size=224,
                 roi_pad=0.25, augment=False, roi_jitter=0.0):
        self.root = root
        self.work_size, self.seg_size, self.crop_size = work_size, seg_size, crop_size
        self.roi_pad, self.roi_jitter = roi_pad, roi_jitter
        self.aug = train_augmentations() if augment else None

        teeth = pd.read_csv(os.path.join(root, "teeth.csv"), dtype={"image_id": str, "patient_id": str})
        splits = pd.read_csv(os.path.join(root, "splits.csv"), dtype={"image_id": str})
        ids = splits.loc[splits["split"] == split, "image_id"].tolist()
        self.image_ids = sorted(ids)
        self.teeth = {i: g.sort_values("instance_id") for i, g in teeth[teeth.image_id.isin(ids)].groupby("image_id")}
        self.image_ids = [i for i in self.image_ids if i in self.teeth]

    def __len__(self):
        return len(self.image_ids)

    def class_counts(self) -> np.ndarray:
        c = np.zeros(len(CLASSES))
        for g in self.teeth.values():
            for lab in g["label"]:
                c[CLASS_TO_IDX[lab]] += 1
        return c

    def load(self, image_id):
        img = cv2.cvtColor(cv2.imread(find_image(self.root, image_id)), cv2.COLOR_BGR2RGB)
        inst = cv2.imread(os.path.join(self.root, "instances", image_id + ".png"), cv2.IMREAD_UNCHANGED)
        if inst.ndim == 3:
            inst = inst[..., 0]
        img = resize_long_side(img, self.work_size, cv2.INTER_AREA)
        inst = cv2.resize(inst.astype(np.uint16), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return img, inst

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img, inst = self.load(image_id)
        if self.aug is not None:
            out = self.aug(image=img, mask=inst.astype(np.int32))
            img, inst = out["image"], out["mask"].astype(np.uint16)

        seg_img = cv2.resize(img, (self.seg_size, self.seg_size), interpolation=cv2.INTER_AREA)
        seg_mask = cv2.resize((inst > 0).astype(np.uint8), (self.seg_size, self.seg_size),
                              interpolation=cv2.INTER_NEAREST)

        crops, labels, boxes, inst_ids = [], [], [], []
        for _, row in self.teeth[image_id].iterrows():
            box = box_from_mask(inst == int(row["instance_id"]))
            if box is None:  # region rotated out of frame by augmentation
                continue
            crops.append(normalize(crop_roi(img, box, self.roi_pad, self.crop_size, self.roi_jitter)))
            labels.append(CLASS_TO_IDX[row["label"]])
            boxes.append(box)
            inst_ids.append(int(row["instance_id"]))

        return {
            "image_id": image_id,
            "image": normalize(seg_img),
            "mask": torch.from_numpy(seg_mask[None].astype(np.float32)),
            "crops": torch.stack(crops) if crops else torch.zeros(0, 3, self.crop_size, self.crop_size),
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": boxes,
            "instance_ids": inst_ids,
        }


def collate(batch):
    return {
        "image_id": [b["image_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "crops": torch.cat([b["crops"] for b in batch]),
        "labels": torch.cat([b["labels"] for b in batch]),
        "n_teeth": [len(b["labels"]) for b in batch],
        "boxes": [b["boxes"] for b in batch],
        "instance_ids": [b["instance_ids"] for b in batch],
    }
