import cv2
import numpy as np
import json
import sys
import math

IMAGE_PATH = sys.argv[1]
MIN_DISTANCE = 20 
# --- NY KONSTANT ---
MIN_SIZE = 8  # Blobs mindre än 4px i bredd eller höjd ignoreras

def main():
    print("image recoloring")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Could not load image.")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 0, 0])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)    
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    thresh = cv2.dilate(eroded, kernel, iterations=1)

    cv2.imwrite('results\\' + IMAGE_PATH + 'haiku3thresh.output.png', thresh)
    
    print("finding all regions")
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw_results = []
    print("extracting detections")
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_SIZE or h < MIN_SIZE:
            continue

        center_x = x + w // 2
        center_y = y + h // 2
        area = cv2.contourArea(contour)
        
        raw_results.append({
            'x': int(center_x),
            'y': int(center_y),
            'width': int(w),
            'height': int(h),
            'area': float(area)
        })

    #raw_results.sort(key=lambda x: x['area'], reverse=True)
    
    filtered_results = []
    for candidate in raw_results:
        keep = True
        for accepted in filtered_results:
            dist = math.sqrt((candidate['x'] - accepted['x'])**2 + 
                             (candidate['y'] - accepted['y'])**2)
            
            if dist < MIN_DISTANCE:
                keep = False
                break
        
        if keep:
            candidate['id'] = len(filtered_results)
            filtered_results.append(candidate)

    print("writes json")
    with open('results\\' + IMAGE_PATH + 'haiku3.json', 'w') as f:
        json.dump(filtered_results, f, indent=4)

    output_img = img.copy()
    for item in filtered_results:
        cv2.circle(output_img, (item['x'], item['y']), 8, (0, 255, 255), 3)
        
    print("writes image")
    cv2.imwrite('results\\' + IMAGE_PATH + 'haiku3.output.png', output_img)
    print(f"Found {len(filtered_results)} unique regions after filtering (Min size: {MIN_SIZE}px)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python program <full_path_to_image.tif>")
        sys.exit(1)
    main()