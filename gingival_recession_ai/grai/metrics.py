"""Evaluation metrics reported in the manuscript."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from .data import CLASSES


def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, eps=1e-7):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    p, g = pred_bin.sum(), gt_bin.sum()
    union = p + g - inter
    if p + g == 0:
        return 1.0, 1.0
    return float(2 * inter / (p + g + eps)), float(inter / (union + eps))


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0)))


def classification_report(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """y_true: (N,) int labels; probs: (N, C) softmax probabilities."""
    from scipy.stats import norm

    y_pred = probs.argmax(1)
    C = len(CLASSES)
    rep = {"n_teeth": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
           "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(C))).tolist(),
           "per_class": {}}
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(C)), zero_division=0)

    aucs, sens_l, spec_l, youden_l = [], [], [], []
    for c, name in enumerate(CLASSES):
        yb = (y_true == c).astype(int)
        pb = (y_pred == c).astype(int)
        tp = int(((pb == 1) & (yb == 1)).sum()); tn = int(((pb == 0) & (yb == 0)).sum())
        fp = int(((pb == 1) & (yb == 0)).sum()); fn = int(((pb == 0) & (yb == 1)).sum())
        d = {"precision": float(prec[c]), "recall_sensitivity": float(rec[c]), "f1": float(f1[c]),
             "support": int(sup[c]),
             "specificity": tn / (tn + fp) if tn + fp else float("nan"),
             "ovr_accuracy": (tp + tn) / len(yb)}
        if 0 < yb.sum() < len(yb):
            auc = roc_auc_score(yb, probs[:, c])
            fpr, tpr, thr = roc_curve(yb, probs[:, c])
            j = tpr - fpr
            k = int(np.argmax(j))
            se = hanley_mcneil_se(auc, yb.sum(), len(yb) - yb.sum())
            z = (auc - 0.5) / se if se > 0 else np.inf
            d.update({"auc": float(auc), "auc_se": se,
                      "auc_95ci": [float(max(0, auc - 1.96 * se)), float(min(1, auc + 1.96 * se))],
                      "auc_p_value_vs_0.5": float(2 * (1 - norm.cdf(abs(z)))),
                      "youden_max": float(j[k]), "youden_threshold": float(thr[k]),
                      "sens_at_youden": float(tpr[k]), "spec_at_youden": float(1 - fpr[k]),
                      "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}})
            aucs.append(auc); sens_l.append(tpr[k]); spec_l.append(1 - fpr[k]); youden_l.append(j[k])
        rep["per_class"][name] = d

    rep["macro_precision"] = float(np.mean(prec))
    rep["macro_sensitivity"] = float(np.mean(rec))
    rep["macro_specificity"] = float(np.nanmean([rep["per_class"][n]["specificity"] for n in CLASSES]))
    rep["macro_f1"] = float(np.mean(f1))
    if aucs:
        rep["macro_auc_ovr"] = float(np.mean(aucs))
        rep["macro_youden"] = float(np.mean(youden_l))
        rep["macro_sens_at_youden"] = float(np.mean(sens_l))
        rep["macro_spec_at_youden"] = float(np.mean(spec_l))
    return rep
