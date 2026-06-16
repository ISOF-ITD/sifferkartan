import numpy as np
from paddleocr import PaddleOCR
import sys
import json

"""
Använder paddleOCR för detektion.

Dålig detektion, enkelt att använda med python.
"""

def extract_digit_positions(image_path):
    # Initiera PaddleOCR
    ocr = PaddleOCR(lang='en')  # Använd engelska som språk

    # Läs bilden
    result = ocr.ocr(image_path)
    resulttwo = []

    # Iterera över resultaten
    for line in result:
        # Line är en lista med [bbox, text, confidence]
        bbox = line[0]
        text = line[1]
        confidence = line[2]

        # Kontrollera om texten är en siffra
        if text.isdigit():
            print(f"Siffra: {text}, Position: {bbox}, Förtroende: {confidence}")
            resulttwo.append({text,bbox,confidence})
        
    # Skriv ut resultatet till en JSON-fil
    with open(IMAGE_PATH+'paddle.json', 'w') as f:
        json.dump(resulttwo, f, indent=4)

# Använd funktionen
IMAGE_PATH   = sys.argv[1]                     # t.ex. C:\Karta\my_map.tif
def main():
    extract_digit_positions(IMAGE_PATH)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python program <full_path_to_image.tif>")
        sys.exit(1)
    main()