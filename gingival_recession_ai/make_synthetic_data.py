"""Generate a small *synthetic* dataset to smoke-test the pipeline before real data is ready.

Images are pink "gingiva" with white "teeth"; each recession region is a yellowish U-shaped
exposure whose depth / interdental shape loosely depends on the RT class. It only checks that
everything runs end-to-end — results on it say nothing about clinical performance.

    python make_synthetic_data.py --out data_synth --n 40
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd


def make(rng, h=480, w=720):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (150, 110, 200)  # BGR pinkish gingiva
    inst = np.zeros((h, w), np.uint16)
    rows, n_teeth = [], 6
    tw = w // (n_teeth + 1)
    k = 0
    for t in range(n_teeth):
        cx = int(tw * (t + 1))
        top = int(h * 0.15)
        cv2.rectangle(img, (cx - tw // 2 + 6, top), (cx + tw // 2 - 6, int(h * 0.55)), (235, 240, 245), -1)
        if rng.random() < 0.55:
            label = rng.choice(["RT1", "RT2", "RT3"], p=[0.28, 0.46, 0.26])
            depth = {"RT1": 0.08, "RT2": 0.14, "RT3": 0.20}[label] * h * rng.uniform(0.8, 1.2)
            width = {"RT1": 0.45, "RT2": 0.6, "RT3": 0.8}[label] * tw
            y0 = int(h * 0.55)
            m = np.zeros((h, w), np.uint8)
            cv2.ellipse(m, (cx, y0), (int(width / 2), int(depth)), 0, 0, 180, 1, -1)
            if label != "RT1":  # interdental loss -> notches at the sides
                for s in (-1, 1):
                    cv2.circle(m, (cx + s * int(width / 2), y0 + 4), int(depth * (0.3 if label == "RT2" else 0.6)), 1, -1)
            k += 1
            inst[m > 0] = k
            img[m > 0] = (120, 200, 230)
            rows.append({"instance_id": k, "label": label})
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img, inst, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "instances"), exist_ok=True)
    all_rows = []
    i = 0
    while i < args.n:
        img, inst, rows = make(rng)
        if not rows:
            continue
        iid = f"img{i:03d}"
        cv2.imwrite(os.path.join(args.out, "images", iid + ".jpg"), img)
        cv2.imwrite(os.path.join(args.out, "instances", iid + ".png"), inst)
        all_rows += [{"image_id": iid, "patient_id": f"p{i:03d}", **r} for r in rows]
        i += 1
    pd.DataFrame(all_rows).to_csv(os.path.join(args.out, "teeth.csv"), index=False)
    print(f"wrote {args.n} images, {len(all_rows)} teeth to {args.out}")


if __name__ == "__main__":
    main()
