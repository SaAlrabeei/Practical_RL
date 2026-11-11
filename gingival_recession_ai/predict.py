"""Run the trained model on new clinical photographs.

    python predict.py --ckpt runs/exp1/best.pt --images path/to/img.jpg [more.jpg ...] --out predictions/
"""
import argparse
import json
import os

import cv2
import torch

from evaluate import load_model
from grai.inference import draw_prediction, predict_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, cfg = load_model(args.ckpt, args.device)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for path in args.images:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        prob, teeth, work = predict_image(model, img, cfg, args.device)
        vis = draw_prediction(work, prob, teeth, cfg["inference"]["mask_threshold"])
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out, name + "_pred.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        results[name] = teeth
        print(name, [(t["label"], round(max(t["probs"]), 3)) for t in teeth])
    json.dump(results, open(os.path.join(args.out, "predictions.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
