#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, json, tqdm, torch, cv2, numpy as np
from ultralytics import YOLO
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

# --------------------  CONFIG  --------------------
IMAGE_PATH   = sys.argv[1]                     # ex: C:\Sökväg\Till\map_sheet.tif
YOLO_MODEL   = "yolov8n.pt"                    # eller full sökväg C:\...\yolov8n.pt
OCR_DIR      = "layoutlmv3_base"               # mappen du skapade ovan
OUTPUT_JSON  = "result.geojson"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------

def pix2geo(x, y, gt):
    """Konvertera pixel‑koordinat till geografisk koordinat.
    Returnerar plain Python-float‑värden."""
    lon = float(gt[0] + x * gt[1] + y * gt[2])
    lat = float(gt[3] + x * gt[4] + y * gt[5])
    return lon, lat

def ocr_region(pil_img, processor, model):
    encoding = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**encoding)
    preds = torch.argmax(outputs.logits, dim=-1)
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
    words = []
    for token, pred in zip(tokens, preds[0]):
        if token.startswith("##"):
            words[-1] = words[-1] + token[2:]
        else:
            words.append(token)
    return " ".join(words).replace("[PAD]", "").strip()

def main():
    # Läs in bilden i full upplösning
    img = cv2.imread(IMAGE_PATH)
    h, w = img.shape[:2]

    # -------------------------------------------------
    # 1) Text‑detektion med YOLOv8 (auto‑GPU/CPU)
    detector = YOLO(YOLO_MODEL)
    results = detector(img, imgsz=10024, conf=0.25)   # 1024 px max‑storlek, justera om du vill
    boxes = results[0].boxes.xyxy.cpu().numpy()    # [[x1,y1,x2,y2], …]

    # -------------------------------------------------
    # 2) Ladda OCR‑modellen
    processor = LayoutLMv3Processor.from_pretrained(OCR_DIR)
    ocr_model = LayoutLMv3ForTokenClassification.from_pretrained(OCR_DIR).to(DEVICE)

    # -------------------------------------------------
    # 3) (valfri) Georeferens – placeholder‑gt
    # Om du har en .tfw‑fil, ersätt med de värden du läser där:
    #   gt = (ulx, xres, rot, uly, rot, -yres)
    gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)   # <-- ENDAST EXEMPEL

    features = []
    for i, (x1, y1, x2, y2) in enumerate(tqdm.tqdm(boxes, desc="Bearbetar boxar")):
        roi = img[int(y1):int(y2), int(x1):int(x2)]
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        text = ocr_region(pil_roi, processor, ocr_model)

        # Centrera boxen för en punkt
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        lon, lat = pix2geo(cx, cy, gt)

        features.append({
            "type": "Feature",
            "properties": {"text": text, "box_id": i},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Klar! Resultatet finns i {OUTPUT_JSON}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python map_ocr.py <full_path_to_image.tif>")
        sys.exit(1)
    main()