"""Patient-level 70/15/15 split that keeps RT1/RT2/RT3 proportions balanced across subsets.

All teeth of a patient (and every image of that patient) go to the same subset -> no leakage.

    python split_dataset.py --data data [--ratios 0.70 0.15 0.15] [--seed 42]
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=2000)
    args = ap.parse_args()

    teeth = pd.read_csv(os.path.join(args.data, "teeth.csv"), dtype={"image_id": str, "patient_id": str})
    per_pat = pd.crosstab(teeth.patient_id, teeth.label).reindex(columns=["RT1", "RT2", "RT3"], fill_value=0)
    pats = per_pat.index.to_numpy()
    counts = per_pat.to_numpy()
    target = counts.sum(0) / counts.sum()
    r = np.array(args.ratios) / sum(args.ratios)
    n_tr = int(round(r[0] * len(pats))); n_va = int(round(r[1] * len(pats)))

    rng = np.random.default_rng(args.seed)
    best, best_assign = np.inf, None
    for _ in range(args.trials):
        perm = rng.permutation(len(pats))
        parts = [perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]]
        cost = 0
        for p, frac in zip(parts, r):
            c = counts[p].sum(0)
            cost += np.abs(c / max(c.sum(), 1) - target).sum() + abs(c.sum() / counts.sum() - frac)
        if cost < best:
            best, best_assign = cost, parts

    pat_split = {}
    for name, idx in zip(["train", "val", "test"], best_assign):
        for i in idx:
            pat_split[pats[i]] = name
    img = teeth.drop_duplicates("image_id")[["image_id", "patient_id"]]
    img["split"] = img.patient_id.map(pat_split)
    img[["image_id", "split"]].to_csv(os.path.join(args.data, "splits.csv"), index=False)

    teeth["split"] = teeth.image_id.map(dict(zip(img.image_id, img.split)))
    tab = pd.crosstab(teeth.split, teeth.label, margins=True, margins_name="Total")
    tab.insert(0, "images", img.groupby("split").size().reindex(tab.index).fillna(len(img)).astype(int))
    print(tab.reindex(["train", "val", "test", "Total"]).to_string())


if __name__ == "__main__":
    main()
