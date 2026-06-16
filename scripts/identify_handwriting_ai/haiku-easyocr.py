import cv2
import numpy as np
import easyocr
import json
import sys

IMAGE_PATH = sys.argv[1]

"""
Detection of handwriting with easyOCR 1

"""

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
    lower_red1 = np.array([0, 29, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 29, 75])
    upper_red2 = np.array([179, 255, 255])

    # Threshold the HSV image to get only red colors
    #mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)    
    mask1 = cv2.inRange(hsv, (0,0,0), (10,255,255))
    mask2 = cv2.inRange(hsv, (10,0,255), (115,255,255))
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(mask1, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Convert to binary
    thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cv2.imwrite('results\\' + IMAGE_PATH + 'threshhaiku.output.png', thresh)
    sys.exit(1)
    # Initialize EasyOCR reader for digits
    print("loading OCR model")
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
    
    print("processing image")
    results = reader.readtext(thresh, detail=1)

    # Extract digits
    result = []
    print("extracting digits")
    for detection in results:
        text = detection[1]
        confidence = detection[2]
        bbox = detection[0]
        
        # Get center coordinates of bounding box
        x = int((bbox[0][0] + bbox[2][0]) / 2)
        y = int((bbox[0][1] + bbox[2][1]) / 2)
        
        # Only keep if it's a digit and confidence is reasonable
        if text.isdigit() and confidence > 0.3:
            result.append({
                'value': text,
                'x': x,
                'y': y,
                'confidence': confidence
            })
    
    # Write results to JSON
    print("writes json")
    with open('results\\' + IMAGE_PATH + 'haiku.json', 'w') as f:
        json.dump(result, f, indent=4)

    # Create output image with detected digits
    output_img = img.copy()
    for item in result:
        cv2.putText(output_img,
                    str(item['value']) + " " + str(item['x']) + ", " + str(item['y']),
                    (item['x'], item['y']),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

    print("writes image")
    cv2.imwrite('results\\' + IMAGE_PATH + 'haiku.output.png', output_img)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python program <full_path_to_image.tif>")
        sys.exit(1)
    main()
