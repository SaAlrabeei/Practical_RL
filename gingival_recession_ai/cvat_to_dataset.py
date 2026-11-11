"""Convert a CVAT export ("CVAT for images 1.1" XML) into the training layout.

Annotation convention in CVAT
-----------------------------
* One **polygon (or mask/brush) per recession-affected tooth**, outlining the exposed
  root / recession region (gingival margin -> CEJ).
* The polygon **label** is the Cairo type: ``RT1``, ``RT2`` or ``RT3``.
* Optional attribute ``tooth`` (e.g. FDI number) is copied to teeth.csv.

Patient IDs: by default each image is its own patient (one image per patient, as in the
study). Supply ``--patients patients.csv`` (columns image_id, patient_id) otherwise.

    python cvat_to_dataset.py --xml annotations.xml --images raw_images/ --out data/
"""
import argparse
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

VALID = {"RT1", "RT2", "RT3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--images", required=True, help="folder with the original photographs")
    ap.add_argument("--out", default="data")
    ap.add_argument("--patients", help="optional CSV image_id,patient_id")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "instances"), exist_ok=True)
    pmap = {}
    if args.patients:
        p = pd.read_csv(args.patients, dtype=str)
        pmap = dict(zip(p.image_id, p.patient_id))

    rows = []
    for im in ET.parse(args.xml).getroot().iter("image"):
        name = im.get("name")
        image_id = os.path.splitext(os.path.basename(name))[0]
        w, h = int(im.get("width")), int(im.get("height"))
        inst = np.zeros((h, w), np.uint16)
        k = 0
        for shape in list(im):
            label = shape.get("label", "").strip().upper()
            if label not in VALID:
                continue
            if shape.tag == "polygon":
                pts = np.array([[float(v) for v in p.split(",")] for p in shape.get("points").split(";")])
                k += 1
                cv2.fillPoly(inst, [np.round(pts).astype(np.int32)], int(k))
            elif shape.tag == "mask":  # CVAT RLE brush mask
                left, top = int(shape.get("left")), int(shape.get("top"))
                mw, mh = int(shape.get("width")), int(shape.get("height"))
                rle = [int(v) for v in shape.get("rle").split(",")]
                flat, val, pos = np.zeros(mw * mh, np.uint8), 0, 0
                for r in rle:
                    flat[pos:pos + r] = val; pos += r; val = 1 - val
                k += 1
                sub = inst[top:top + mh, left:left + mw]
                sub[flat.reshape(mh, mw) > 0] = k
            else:
                continue
            attrs = {a.get("name"): a.text for a in shape.findall("attribute")}
            rows.append({"image_id": image_id, "patient_id": pmap.get(image_id, image_id),
                         "instance_id": k, "label": label, "tooth": attrs.get("tooth", "")})
        if k == 0:
            print("warning: no RT polygons in", name)
            continue
        src = os.path.join(args.images, os.path.basename(name))
        if not os.path.exists(src):
            src = os.path.join(args.images, name)
        shutil.copy(src, os.path.join(args.out, "images", image_id + os.path.splitext(name)[1].lower()))
        cv2.imwrite(os.path.join(args.out, "instances", image_id + ".png"), inst)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "teeth.csv"), index=False)
    print(f"{df.image_id.nunique()} images, {len(df)} teeth")
    print(df.label.value_counts().sort_index().to_string())
    print("next: python split_dataset.py --data", args.out)


if __name__ == "__main__":
    main()
