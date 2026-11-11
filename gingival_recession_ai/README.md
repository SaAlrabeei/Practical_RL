# Gingival Recession AI: segmentation plus Cairo RT1/RT2/RT3 classification

This is a PyTorch implementation of the multi-task deep-learning model described in the
manuscript *"Development of an Artificial Intelligence–Based Clinical Image Model for
Detection and Classification of Anterior Gingival Recession"*.

```
clinical photo ──► preprocessing (normalize, resize 256×256)
                     │
                     ▼
           EfficientNet-B4 encoder (ImageNet, shared)
                     │
         ┌───────────┴──────────────────────────┐
         ▼                                       │
   U-Net decoder ──► binary recession mask       │
         │                                       │
         ▼                                       │
   connected components ──► per-tooth ROI crops ─┘ (re-encoded by the same encoder)
                                                 ▼
               AvgPool → BN → Dropout → FC256 → ReLU → Dropout → FC3
                                                 ▼
                                  RT1 / RT2 / RT3 for each tooth
```

| Item (manuscript)        | Implementation                                                          |
|--------------------------|-------------------------------------------------------------------------|
| Encoder / decoder        | `segmentation_models_pytorch.Unet("efficientnet-b4", "imagenet")`       |
| Classification branch    | deepest encoder features of each tooth crop → pool/BN/dropout/FC256/FC3 |
| Segmentation loss        | 0.5 · Dice + 0.5 · BCEWithLogits                                        |
| Classification loss      | class-weighted cross-entropy (inverse frequency)                        |
| Total loss               | seg + 0.8 · cls                                                         |
| Optimiser                | AdamW + ReduceLROnPlateau                                               |
| Epochs / early stopping  | max 40, stop after 10 epochs without val-total-loss improvement         |
| Batch size               | 2                                                                        |
| Checkpoint selection     | lowest **validation total loss**                                        |
| Augmentation (train only)| flip, rotate ±15°, elastic, hue/saturation, Gaussian noise              |
| Split                    | patient-level 70/15/15, class-balanced (`split_dataset.py`)             |
| Metrics                  | accuracy, precision, sensitivity, specificity, F1, OvR ROC-AUC (+SE, 95% CI, p), macro AUC, max Youden index & threshold, Cohen's κ, Dice, IoU |

All hyper-parameters are in `config.yaml`.

## 1. Install

```bash
cd gingival_recession_ai
pip install -r requirements.txt     # use a CUDA build of torch if you have a GPU
```

## 2. Annotate in CVAT

For **each recession-affected tooth**, draw **one polygon** (or brush mask) around the
recession region, from the gingival margin to the CEJ. Set its **label** to `RT1`, `RT2` or
`RT3`. You can also add a `tooth` attribute, such as an FDI number.
Then export with **"CVAT for images 1.1"**.

```bash
python cvat_to_dataset.py --xml annotations.xml --images raw_photos/ --out data
#   add --patients patients.csv (image_id,patient_id) if a patient has more than one image
python split_dataset.py --data data          # writes data/splits.csv and prints a Table-2-style summary
```

The dataset layout is:

```
data/images/<id>.jpg   data/instances/<id>.png (0 = bg, k = tooth k)   data/teeth.csv   data/splits.csv
```

## 3. Train, evaluate and predict

```bash
python train.py --config config.yaml --data data --out runs/exp1
python evaluate.py --ckpt runs/exp1/best.pt --split test --roi-source gt     # per-tooth classification (paper tables)
python evaluate.py --ckpt runs/exp1/best.pt --split test --roi-source pred   # fully automatic pipeline
python predict.py  --ckpt runs/exp1/best.pt --images new_photo.jpg --out predictions/
```

`evaluate.py` writes `metrics.json`, `predictions.npz` and `roc_confusion.png`, which holds the
ROC curves and the confusion matrix. `train.py` writes `best.pt`, `history.json` and
`training_curves.png`.

## Quick smoke test (no real data)

```bash
python make_synthetic_data.py --out data_synth --n 40
python split_dataset.py --data data_synth
python train.py --data data_synth --out runs/synth --epochs 3 --no-pretrained
python evaluate.py --ckpt runs/synth/best.pt --split test
```

## Notes and design choices

- **Teacher forcing for ROIs.** During training, the per-tooth crops come from the
  *ground-truth* instance masks, with random box jitter (`roi_jitter`). The jitter makes the
  classifier robust to imperfect predicted masks. At inference, the ROIs come from the
  connected components of the *predicted* mask.
- **Crops come from a higher-resolution image.** Segmentation runs at 256×256 as in the
  paper. Tooth crops are taken from the image resized to `work_size` = 768 px on the long
  side, so fine CEJ and interdental detail is kept.
- **Neighbouring teeth.** If recession regions on two neighbouring teeth touch, the predicted
  mask can merge them into one component. `--roi-source pred` reports these teeth as
  missed. If this happens often, add a separate tooth-instance segmentation step.
- **Batch size 2 and BatchNorm.** This setting matches the paper. With a bigger GPU, a larger
  batch usually gives more stable BatchNorm statistics.
- **Statistics.** The p-values and 95% CIs for AUC use the Hanley–McNeil standard error with
  an asymptotic normal test. SPSS uses a nonparametric standard error, so its values can
  differ slightly. To report results from SPSS, export `predictions.npz` to it.
