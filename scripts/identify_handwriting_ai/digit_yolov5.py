#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, json, torch, cv2, tqdm
from pathlib import Path
from yolov5 import YOLOv5

IMAGE_PATH  = sys.argv[1]
OUTPUT_JSON = "digits_yolov5.geojson"

# -------------------------------------------------
# Ladda YOLOv5‑modellen (CPU/GPU automatiskt)
model_path = Path("yolov5s-digit.pt")
yolo = YOLOv5(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

# Läs bilden
orig = cv2.imread(IMAGE_PATH)

# YOLOv5‑inference (standard‑sized 640, men vi behåller original för geo‑räkning)
results = yolo.predict(orig, size=640, conf=0.3, iou=0.4)

# Resultatet är en lista med dicts: {'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'}
detections = results.pandas().xyxy[0]   # Pandas‑DataFrame

# Georeferens placeholder (byt ut mot riktig gt)
gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

def pix2geo(x, y, gt):
    return (float(gt[0] + x*gt[1] + y*gt[2]),
            float(gt[3] + x*gt[4] + y*gt[5]))

features = []
for _, row in tqdm.tqdm(detections.iterrows(), total=len(detections), desc="YOLOv5‑siffra"):
    # Vi filtrerar på klassnamnet – modellen har bara en klass "digit"
    if row['name'] != 'digit':
        continue
    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    w, h = x2 - x1, y2 - y1
    # Centrera
    cx, cy = x1 + w/2.0, y1 + h/2.0
    lon, lat = pix2geo(cx, cy, gt)

    # Eftersom YOLO‑modellen redan har klassificerat siffran, kan vi läsa av den från namnet
    digit = row['name']   # runt "0"‑"9" – i detta fin‑tuned‑repo är namnet själva siffran

    features.append({
        "type": "Feature",
        "properties": {"digit": digit, "bbox": [int(x1), int(y1), int(w), int(h)]},
        "geometry": {"type": "Point", "coordinates": [lon, lat]}
    })

geojson = {"type": "FeatureCollection", "features": features}
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print(f"\n✅ Klar! {len(features)} siffror sparade i {OUTPUT_JSON}")
