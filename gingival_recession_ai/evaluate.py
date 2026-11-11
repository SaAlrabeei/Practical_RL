"""Evaluate a checkpoint on the held-out test set.

    python evaluate.py --ckpt runs/exp1/best.pt --split test [--roi-source gt|pred]

--roi-source gt   : classify each annotated tooth ROI (isolates classification quality;
                    this is the per-tooth evaluation reported in the manuscript tables)
--roi-source pred : fully automatic - ROIs come from the predicted mask and are matched
                    to annotated teeth by box IoU; unmatched teeth are reported as missed
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from grai.data import CLASS_TO_IDX, CLASSES, GingivalRecessionDataset, box_from_mask
from grai.inference import box_iou, predict_image
from grai.metrics import classification_report, dice_iou
from grai.model import MultiTaskModel


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ck["cfg"]["model"]
    model = MultiTaskModel(m["encoder"], None, m["num_classes"], m["cls_hidden"], m["dropout"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["cfg"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--roi-source", choices=["gt", "pred"], default="gt")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.device)
    d = cfg["data"]
    if args.data: d["root"] = args.data
    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}_{args.roi_source}")
    os.makedirs(out_dir, exist_ok=True)
    ds = GingivalRecessionDataset(d["root"], args.split, d["work_size"], d["seg_size"], d["crop_size"], d["roi_pad"])

    y_true, probs, dices, ious, missed, false_pos = [], [], [], [], 0, 0
    for i in range(len(ds)):
        s = ds[i]
        logits = model.segment(s["image"][None].to(args.device))
        pm = (torch.sigmoid(logits)[0, 0] > cfg["inference"]["mask_threshold"]).cpu().numpy()
        dsc, iou = dice_iou(pm, s["mask"][0].numpy() > 0.5); dices.append(dsc); ious.append(iou)

        if args.roi_source == "gt":
            if len(s["labels"]):
                p = torch.softmax(model.classify(s["crops"].to(args.device)), 1).cpu().numpy()
                y_true += s["labels"].tolist(); probs += list(p)
        else:
            img, inst = ds.load(s["image_id"])
            _, teeth, _ = predict_image(model, img, cfg, args.device)
            gt = [(box_from_mask(inst == iid), CLASS_TO_IDX[lab]) for iid, lab in
                  zip(ds.teeth[s["image_id"]]["instance_id"], ds.teeth[s["image_id"]]["label"])]
            used = set()
            for gbox, glab in gt:
                if gbox is None:
                    continue
                best, bj = 0, -1
                for j, t in enumerate(teeth):
                    if j in used: continue
                    v = box_iou(gbox, t["box"])
                    if v > best: best, bj = v, j
                if bj >= 0 and best >= cfg["inference"]["match_iou"]:
                    used.add(bj); y_true.append(glab); probs.append(np.array(teeth[bj]["probs"]))
                else:
                    missed += 1
            false_pos += len(teeth) - len(used)

    y_true, probs = np.array(y_true), np.array(probs)
    rep = classification_report(y_true, probs) if len(y_true) else {}
    rep["segmentation"] = {"mean_dice": float(np.mean(dices)), "mean_iou": float(np.mean(ious)), "n_images": len(ds)}
    if args.roi_source == "pred":
        rep["detection"] = {"matched_teeth": int(len(y_true)), "missed_teeth": missed, "unmatched_predictions": false_pos}
    json.dump(rep, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, "predictions.npz"), y_true=y_true, probs=probs)

    print(f"\n=== {args.split} | ROI source: {args.roi_source} ===")
    print(f"Segmentation: Dice {rep['segmentation']['mean_dice']:.3f}  IoU {rep['segmentation']['mean_iou']:.3f}")
    if "detection" in rep: print("Detection:", rep["detection"])
    if len(y_true):
        print(f"Teeth: {rep['n_teeth']}  Accuracy {rep['accuracy']:.3f}  Cohen kappa {rep['cohen_kappa']:.3f}")
        print(f"{'RT':5}{'Prec':>7}{'Sens':>7}{'Spec':>7}{'F1':>7}{'AUC':>7}{'Youden':>8}")
        for c in CLASSES:
            r = rep["per_class"][c]
            print(f"{c:5}{r['precision']:7.3f}{r['recall_sensitivity']:7.3f}{r['specificity']:7.3f}{r['f1']:7.3f}"
                  f"{r.get('auc', float('nan')):7.3f}{r.get('youden_max', float('nan')):8.3f}")
        if "macro_auc_ovr" in rep:
            print(f"Macro AUC (OvR) {rep['macro_auc_ovr']:.3f}  macro Youden {rep['macro_youden']:.3f}")
        print("Confusion matrix (rows = true RT1..RT3):", rep["confusion_matrix"])
        plot(rep, out_dir)
    print("saved to", out_dir)


def plot(rep, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for c in CLASSES:
        r = rep["per_class"][c]
        if "roc" in r:
            ax[0].plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{c} (AUC={r['auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity"); ax[0].set_title("ROC (one-vs-rest)"); ax[0].legend()
    cm = np.array(rep["confusion_matrix"])
    ax[1].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax[1].text(j, i, cm[i, j], ha="center", va="center")
    ax[1].set_xticks(range(3), CLASSES); ax[1].set_yticks(range(3), CLASSES)
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true"); ax[1].set_title("confusion matrix")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc_confusion.png"), dpi=130)


if __name__ == "__main__":
    main()
