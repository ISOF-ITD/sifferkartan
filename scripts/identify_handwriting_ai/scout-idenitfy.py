# Kör en gång för att ladda ner moddellen
#import tensorflow as tf

#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

import cv2
import numpy as np
import pytesseract
import json
import sys

"""
Använder tesseractOCR och pytesseract for att undersöka bilden och leta handskriven text

Försök 1. Se scout-identify3

"""


IMAGE_PATH   = sys.argv[1]                     # t.ex. C:\Karta\my_map.tif
def main():
    print("image recoloring")
    # Ladda ner bildfilen
    img = cv2.imread(IMAGE_PATH)

    # Konvertera bilden till gråskala
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #cv2.imwrite('results\\'+IMAGE_PATH+'thresh.output.png', gray)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the range of red color
    lower_red1 = np.array([0, 0, 0])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)    
    #mask1 = cv2.inRange(hsv, (0,0,0), (10,255,255))
    #mask2 = cv2.inRange(hsv, (10,0,255), (115,255,255))
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(mask1, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Convert to binary
    thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite('results\\'+IMAGE_PATH+'thresh.output.png', thresh)
    sys.exit(1) # testa att ändra upplösning o sktiititl
    

    # Använd Tesseract-OCR för att läsa ut text från bilden
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789"'
    #custom_config = r'--oem 3 --psm 12 -c tessedit_char_whitelist="0123456789"'
    print("processing image")
    text = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config=custom_config)
    #text = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    #text = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    #text = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config)

    # Skapa en lista för att lagra resultatet
    result = []

    # Loopa igenom texten och extrahera siffror
    print("extracting digits")
    for i in range(len(text['text'])):
        if text['text'][i].isdigit():
            result.append({
                'value': text['text'][i],
                'x': text['left'][i],
                'y': text['top'][i]
            })
    # Skriv ut resultatet till en JSON-fil
    print("writes json")
    with open('results\\'+IMAGE_PATH+'sc-ident.json', 'w') as f:
        json.dump(result, f, indent=4)
        #json.dump(text, f, indent=4)


    #sys.exit(1)

    # Skapa en ny bild med punkter på varje instans av de hittade nummerna
    output_img = img.copy()
    for item in result:
        #cv2.circle(output_img, (item['x'], item['y']), 5, (0, 255, 0), -1)
        cv2.putText(output_img,
                    str(item['value']) + " " + str(item['x']) + ", " + str(item['y']),
                    (item['x'], item['y']),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

    # Spara den nya bilden
    print("writes image")
    cv2.imwrite('results\\'+IMAGE_PATH+'sc-ident.output.png', output_img)

# Stänger av programmet om ingen bild specificeras
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python program <full_path_to_image.tif>")
        sys.exit(1)
    main()