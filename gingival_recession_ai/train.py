"""Train the multi-task gingival-recession model.

    python train.py --config config.yaml [--data data] [--out runs/exp1] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from grai.data import CLASSES, GingivalRecessionDataset, collate
from grai.metrics import dice_iou
from grai.model import MultiTaskLoss, MultiTaskModel


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_loader(cfg, split, train):
    d, t = cfg["data"], cfg["train"]
    ds = GingivalRecessionDataset(d["root"], split, d["work_size"], d["seg_size"], d["crop_size"],
                                  d["roi_pad"], augment=train, roi_jitter=t["roi_jitter"] if train else 0.0)
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=train, num_workers=d["num_workers"],
                    collate_fn=collate, drop_last=False)
    return ds, dl


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot = {"loss": 0.0, "seg": 0.0, "cls": 0.0, "n": 0}
    dices, ious, correct, nteeth = [], [], 0, 0
    with torch.set_grad_enabled(train):
        for b in tqdm(loader, leave=False, desc="train" if train else "val"):
            img, mask = b["image"].to(device), b["mask"].to(device)
            crops, labels = b["crops"].to(device), b["labels"].to(device)
            seg_logits, cls_logits = model(img, crops)
            loss, seg_l, cls_l = loss_fn(seg_logits, mask, cls_logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n = img.shape[0]
            tot["loss"] += loss.item() * n; tot["seg"] += seg_l.item() * n; tot["cls"] += cls_l.item() * n
            tot["n"] += n
            pred = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
            for p, g in zip(pred, mask.cpu().numpy() > 0.5):
                dsc, iou = dice_iou(p, g); dices.append(dsc); ious.append(iou)
            if labels.numel():
                correct += (cls_logits.argmax(1) == labels).sum().item(); nteeth += labels.numel()
    n = max(tot.pop("n"), 1)
    out = {k: v / n for k, v in tot.items()}
    out.update(dice=float(np.mean(dices)), iou=float(np.mean(ious)), cls_acc=correct / max(nteeth, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data: cfg["data"]["root"] = args.data
    if args.out: cfg["train"]["out_dir"] = args.out
    if args.epochs: cfg["train"]["epochs"] = args.epochs
    if args.no_pretrained: cfg["model"]["encoder_weights"] = None
    t = cfg["train"]
    set_seed(t["seed"])
    os.makedirs(t["out_dir"], exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(t["out_dir"], "config.yaml"), "w"))

    train_ds, train_dl = make_loader(cfg, "train", True)
    val_ds, val_dl = make_loader(cfg, "val", False)
    counts = train_ds.class_counts()
    print(f"train images={len(train_ds)} teeth={dict(zip(CLASSES, counts.astype(int)))} | val images={len(val_ds)}")

    # inverse-frequency class weights (normalised to mean 1) for the class-weighted CE
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    class_weights = torch.tensor(w / w.mean(), dtype=torch.float32)
    print("class weights:", class_weights.numpy().round(3))

    m = cfg["model"]
    model = MultiTaskModel(m["encoder"], m["encoder_weights"], m["num_classes"], m["cls_hidden"], m["dropout"]).to(args.device)
    loss_fn = MultiTaskLoss(class_weights, t["dice_weight"], t["bce_weight"], t["cls_weight"]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=t["lr_factor"], patience=t["lr_patience"])

    best, bad, history = float("inf"), 0, []
    for epoch in range(1, t["epochs"] + 1):
        tr = run_epoch(model, train_dl, loss_fn, args.device, opt)
        va = run_epoch(model, val_dl, loss_fn, args.device)
        sched.step(va["loss"])
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        print(f"[{epoch:02d}] train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} "
              f"dice {va['dice']:.3f} iou {va['iou']:.3f} cls_acc {va['cls_acc']:.3f}")
        # checkpoint selection on lowest validation TOTAL loss (not IoU), as in the manuscript
        if va["loss"] < best:
            best, bad = va["loss"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": va},
                       os.path.join(t["out_dir"], "best.pt"))
            print(f"     saved best checkpoint (epoch {epoch})")
        else:
            bad += 1
            if bad >= t["early_stop_patience"]:
                print(f"early stopping at epoch {epoch}")
                break
        json.dump(history, open(os.path.join(t["out_dir"], "history.json"), "w"), indent=1)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(ep, [h["train_loss"] for h in history], label="train"); ax[0].plot(ep, [h["val_loss"] for h in history], label="val"); ax[0].set_title("total loss"); ax[0].legend()
        ax[1].plot(ep, [h["val_dice"] for h in history], label="Dice"); ax[1].plot(ep, [h["val_iou"] for h in history], label="IoU"); ax[1].set_title("val segmentation"); ax[1].legend()
        ax[2].plot(ep, [h["val_cls_acc"] for h in history]); ax[2].set_title("val tooth accuracy")
        for a in ax: a.set_xlabel("epoch")
        fig.tight_layout(); fig.savefig(os.path.join(t["out_dir"], "training_curves.png"), dpi=120)
    except Exception as e:  # plotting is optional
        print("plot failed:", e)


if __name__ == "__main__":
    main()
