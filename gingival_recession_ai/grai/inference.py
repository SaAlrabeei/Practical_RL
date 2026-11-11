"""End-to-end inference: image -> recession mask -> tooth ROIs -> RT1/RT2/RT3 per tooth."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .data import CLASSES, crop_roi, normalize, resize_long_side


def rois_from_mask(prob_map: np.ndarray, threshold=0.5, min_area=150):
    """Connected components of the thresholded mask -> list of (box, component_mask)."""
    binary = (prob_map > threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append(([int(x), int(y), int(x + w), int(y + h)], lab == i))
    return out


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def predict_image(model, img_rgb: np.ndarray, cfg: dict, device="cpu"):
    """Returns (prob_map at work size, list of dicts with box/label/probs, work-size image)."""
    d, inf = cfg["data"], cfg["inference"]
    model.eval()
    img = resize_long_side(img_rgb, d["work_size"], cv2.INTER_AREA)
    h, w = img.shape[:2]
    seg_in = normalize(cv2.resize(img, (d["seg_size"], d["seg_size"]), interpolation=cv2.INTER_AREA))
    logits = model.segment(seg_in[None].to(device))
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    rois = rois_from_mask(prob, inf["mask_threshold"], inf["min_component_area"])
    teeth = []
    if rois:
        crops = torch.stack([normalize(crop_roi(img, b, d["roi_pad"], d["crop_size"])) for b, _ in rois])
        p = torch.softmax(model.classify(crops.to(device)), 1).cpu().numpy()
        for (box, _), pr in zip(rois, p):
            teeth.append({"box": box, "label": CLASSES[int(pr.argmax())], "probs": pr.tolist()})
    return prob, teeth, img


def draw_prediction(img_rgb, prob, teeth, threshold=0.5):
    vis = img_rgb.copy()
    overlay = vis.copy()
    overlay[prob > threshold] = (255, 0, 255)
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
    colors = {"RT1": (0, 200, 0), "RT2": (255, 165, 0), "RT3": (255, 0, 0)}
    for t in teeth:
        x1, y1, x2, y2 = t["box"]
        c = colors[t["label"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        txt = f'{t["label"]} {max(t["probs"]):.2f}'
        cv2.putText(vis, txt, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return vis
