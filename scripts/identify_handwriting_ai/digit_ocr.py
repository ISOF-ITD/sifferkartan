#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, json, tqdm, torch, cv2, numpy as np
from ultralytics import YOLO
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

# --------------------  KONFIGURATION --------------------
IMAGE_PATH   = sys.argv[1]                     # t.ex. C:\Karta\my_map.tif
YOLO_MODEL   = "yolov8n-digit.pt"              # handskriven‑siffra‑detektor
OCR_DIR      = "layoutlmv3_base"               # OCR‑modell (kan också avaktiveras, se nedan)
OUTPUT_JSON  = "digits.geojson"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------

"""
Använder ocr-modellen yolov8n.


Använts mest för test av processeringen.

"""


def pix2geo(x, y, gt):
    """Konvertera pixel‑till‑värld – byt ut mot din egen geotransform."""
    return (float(gt[0] + x*gt[1] + y*gt[2]),
            float(gt[3] + x*gt[4] + y*gt[5]))

def ocr_digit(pil_img, processor, model):
    """Avläser en enskild siffra med LayoutLMv3 (behåller kontext av omgivning)."""
    enc = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outs = model(**enc)
    pred = torch.argmax(outs.logits, dim=-1)
    token = processor.tokenizer.convert_ids_to_tokens(enc.input_ids[0])[0]
    # Oftast är token bara en siffra; om modell ger en hel rad, ta första token
    return token.replace("[PAD]", "").strip()

def main():
    # -------------------------------------------------
    # Läs in bild
    img = cv2.imread(IMAGE_PATH)
    h, w = img.shape[:2]

    # -------------------------------------------------
    # 1) Detektera alla handskrivna siffror
    detector = YOLO(YOLO_MODEL)
    # Vi kör med en hög upplösning så att små siffror fångas
    results = detector(img, imgsz=5000, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy()   # [[x1,y1,x2,y2], …]

    # -------------------------------------------------
    # (Valfritt) Ladda OCR‑modell – om du vill ha exakt teckenform
    # Om du är nöjd med bara box‑positionen kan du hoppa över OCR‑delen.
    processor = LayoutLMv3Processor.from_pretrained(OCR_DIR)
    ocr_model = LayoutLMv3ForTokenClassification.from_pretrained(OCR_DIR).to(DEVICE)

    # -------------------------------------------------
    # Georeferens – placeholder (byt mot egen .tfw‑gt)
    gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

    features = []
    for i, (x1, y1, x2, y2) in enumerate(tqdm.tqdm(boxes, desc="Siffra‑detektion")):
        # Klipp ut ROI (lite marginal så att hela siffran med lite bakgrund)
        margin = 4
        xx1, yy1 = max(0, int(x1)-margin), max(0, int(y1)-margin)
        xx2, yy2 = min(w, int(x2)+margin), min(h, int(y2)+margin)
        roi = img[yy1:yy2, xx1:xx2]

        # För OCR: konvertera till PIL och kör LayoutLMv3
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        digit = ocr_digit(pil_roi, processor, ocr_model)

        # Centrera för punkt
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        lon, lat = pix2geo(cx, cy, gt)

        features.append({
            "type": "Feature",
            "properties": {"digit": digit, "box_id": i},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Klar! {len(features)} siffror sparade i {OUTPUT_JSON}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python digit_ocr.py <full_path_to_image.tif>")
        sys.exit(1)
    main()
