import numpy as np
import sys, json, cv2, codecs

from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from collections import defaultdict

INPUT_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
JSON_OUTPUT = [] # append per action taken for each image
global_colors = []
global_failed_colors = []
failed_processing = []

## TODO:
## Check ratio of image-cut, it should be close to 1:1
## Change blur/edge-detection for the different map-types(colors)
## Add recursion/better error handling if the image isnt up to scruff.


## Add way to specify the amount of blur or multiple runs with different amounts of blur.

## when in doubt change the blur - xx

def get_dominant_color(image_path):
    """
    Get the most dominant color in an image.
    Used for classifying what type of map cut technique to use.
        tuple: RGB color tuple (r, g, b)
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        string: the dominant color
    """
    # Open the image
    img = Image.open(image_path)
    
    # Resize for faster processing (optional but recommended)
    img = img.resize((150, 150))
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Reshape to 2D array (pixels × 3 color channels)
    pixels = img_array.reshape(-1, 3)
    
    # Find the most common color
    dominant_color = pixels.mean(axis=0).astype(int)
    dominant = tuple(dominant_color)
    #print(f"Dominant color (RGB): {dominant[0]},{dominant[1]},{dominant[2]}")
    color = ""
    if dominant[0] > 200 and dominant[1] > 180:
    #    print("pink dominant")
        color = "Pink"
        global_colors.append('Pink')
    elif 170 < dominant[0] < 200 and dominant[1] < 200:
    #    print("green dominant")
        color = "Green"
        global_colors.append('Green')
    elif 140 < dominant[0] < 180 and 140 < dominant[1] < 200:
    #    print("darker green dominant")
        color = "Dark green"
        global_colors.append('Dark green')
    else:
        print("unknown color range, check image")
        print(f"Dominant color (RGB): {dominant[0]},{dominant[1]},{dominant[2]}")
        print(f"{image_path}")
        sys.exit(1)
    
    return color

def find_largest_rectangle(image: np.ndarray, image_name:str, edges) -> Optional[Tuple[int, int, int, int]]:
    # Used to find the largest rectangle in the image
    max_area = 0
    rectangle = None
    image2 = image.copy()
    contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 255), 2)
            if area > max_area:
                max_area = area
                rectangle = approx
 
    if rectangle is not None:
        x, y, w, h = cv2.boundingRect(rectangle)
        if (h < 2000 or w < 2000) or not (0.95 < (h / w) < 1.05):# 4000px for tif, 2000px for jpg. Look if 1:1
            cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.imwrite(OUTPUT_FOLDER + f'\\others\\rect-{image_name}.jpg', image2)
            #print(f"No outer rectangle found for {image_name}")
            #print("")
            #JSON_OUTPUT.append({
            #    'name': image_name,
            #    'status': 'NO SUITABLE OUTER RECTANGLE'})
        else:
            cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #cv2.imwrite(OUTPUT_FOLDER + f'\\others\\{image_name}rect.jpg', image2)
            update_output_json('outer', image_name, x, y, x+w,y+h)
            #JSON_OUTPUT.append({
            #    'name':image_name,
            #    'data':{
            #        'outer':{
            #            'coords':[x, y, x +w, y + h]
            #        }
            #    }
            #})
            return (x+10,y+10,x+w-10,y+h-10)
    return

def detect_outer_frame(image: np.ndarray, image_name: str, image_color: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the four corners of a map by finding the corner marks.
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) coordinates, or None if detection fails
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image_color == "Pink":
        #TODO :
        # edit this so that images with pink background get cut correctly.
        # maybe use another techinque
        #blur level matters allllot 
        # implement different levels of it and checks if the image is likley the wrong size, if it is change the blur level. 
        # the inner frame seems to like more blur, the outer less.
   

        # Apply slight blur to reduce noise
        blurred = gray
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #blurred = cv2.blur(blurred,(5,5))
        #blurred = cv2.blur(blurred,(5,5))

     
        # Detect edges (corner marks are typically high-contrast)
        edges = cv2.Canny(blurred, 50, 120)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=200, maxLineGap=0)
    elif image_color == "Green" or image_color == "Dark green":
        # Apply slight blur to reduce noise
        blurred = gray
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred = cv2.blur(blurred,(5,5))
        #blurred = cv2.blur(blurred,(5,5))
        blurred = cv2.blur(blurred,(3,3))

        # Detect edges (corner marks are typically high-contrast)
        edges = cv2.Canny(blurred, 50, 120)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=200, maxLineGap=0)
    else:
        print(f"map-type unknown. skipping {image_name}")
        return
    
    rectangle = find_largest_rectangle(image,image_name,edges)
    if rectangle is not None:
        return rectangle
    
    if lines is None:
        cv2.imwrite(OUTPUT_FOLDER + f'\\others\\edges-{image_name}.jpg', edges)
        return None
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Horizontal lines: ~0° or ~180°
        if angle < 10 or angle > 170:
            horizontal_lines.append((min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)))
        # Vertical lines: ~90°
        elif 80 < angle < 100:
            vertical_lines.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))

    if not horizontal_lines or not vertical_lines:
        cv2.imwrite(OUTPUT_FOLDER + f'\\others\\edges-{image_name}.jpg', edges)
        return None
    
    # Find bounding box from detected lines
    y_coords = [line[0] for line in horizontal_lines] + [line[1] for line in horizontal_lines]
    x_coords = [line[0] for line in vertical_lines] + [line[1] for line in vertical_lines]
   
    if not x_coords or not y_coords:
        cv2.imwrite(OUTPUT_FOLDER + f'\\others\\edges-{image_name}.jpg', edges)
        return None
    
    # Use clustering to find the actual map boundaries (reject outliers)
    y_sorted = sorted(y_coords)
    x_sorted = sorted(x_coords)
    
    y_min = y_sorted[len(y_sorted) // 4]  # Lower quartile
    y_max = y_sorted[3 * len(y_sorted) // 4]  # Upper quartile
    x_min = x_sorted[len(x_sorted) // 4]
    x_max = x_sorted[3 * len(x_sorted) // 4]

    #if smaller than this its cut wrongly.
    if not (0.95 < ((y_max - y_min) / (x_max - x_min + .00001)) < 1.05): #for 1:1 maps
    #if y_max - y_min < 2000 or x_max - x_min < 2000:
        cv2.imwrite(OUTPUT_FOLDER + f'\\others\\edges-{image_name}.jpg', edges)
        return
    update_output_json('outer', image_name, x_min,y_min,x_max,y_max)
    return (x_min, y_min, x_max, y_max)

def detect_inner_frame(image: np.ndarray, image_name:str, image_color: str) -> Optional[Tuple[int, int, int, int]]:
     # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image_color == "Pink":
        # Apply slight blur to reduce noise
        blurred = gray
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #blurred = cv2.blur(blurred,(5,5))
        #blurred = cv2.blur(blurred,(5,5))
        # Detect edges (corner marks are typically high-contrast)
        edges = cv2.Canny(blurred, 50, 120)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=200, maxLineGap=4000)
    elif image_color == "Green" or image_color == "Dark green":
        # Apply slight blur to reduce noise
        blurred = gray
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred = cv2.blur(blurred,(5,5))
        #blurred = cv2.blur(blurred,(5,5))
        blurred = cv2.blur(blurred,(5,5))

        # Detect edges (corner marks are typically high-contrast)
        edges = cv2.Canny(blurred, 50, 120)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=200, maxLineGap=4000)
    else:
        return

    if lines is None:
        print("lines are none")
        cv2.imwrite(OUTPUT_FOLDER + f'\\others\\egdes2-{image_name}.jpg', edges)
        return None
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Horizontal lines: ~0° or ~180°
        if angle < 10 or angle > 170:
            horizontal_lines.append((min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)))
        # Vertical lines: ~90°
        elif 80 < angle < 100:
            vertical_lines.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))

    if not horizontal_lines or not vertical_lines:
        print("hori or vert lines are none")
        cv2.imwrite(OUTPUT_FOLDER + f'\\others\\egdes2-{image_name}.jpg', edges)
        return None
    
    # Find bounding box from detected lines
    y_coords = [line[0] for line in horizontal_lines] + [line[1] for line in horizontal_lines]
    x_coords = [line[0] for line in vertical_lines] + [line[1] for line in vertical_lines]
   
    if not x_coords or not y_coords:
        print("x,y cords are none")
        cv2.imwrite(OUTPUT_FOLDER + f'\\others\\egdes2-{image_name}.jpg', edges)
        return None
    
    # Use clustering to find the actual map boundaries (reject outliers)
    y_sorted = sorted(y_coords)
    x_sorted = sorted(x_coords)
    
    y_min = y_sorted[len(y_sorted) // 4]  # Lower quartile
    y_max = y_sorted[3 * len(y_sorted) // 4]  # Upper quartile
    x_min = x_sorted[len(x_sorted) // 4]
    x_max = x_sorted[3 * len(x_sorted) // 4]

    #if smaller than this its cut wrongly.
    if not (0.95 < ((y_max - y_min) / (x_max - x_min + .00001)) < 1.05): #for 1:1 maps
    #if y_max - y_min < 1400 or x_max - x_min < 1400:
        return 1
    
    update_output_json('inner', image_name, x_min,y_min,x_max,y_max)
    return (x_min, y_min, x_max, y_max)
    
def update_output_json(type:str, image_name:str, x_min:int,y_min:int,x_max:int,y_max:int):
    for item in JSON_OUTPUT:
        if item['name'] == image_name:
            item['data'].update({
                type: {
                    'coords':[int(x_min), int(y_min), int(x_max), int(y_max)]
                }
            })
            item['status'] = type 
            break
    return

def crop_map_image(image_path: str, image_name: str, output_path, padding: int = 0) -> int:
    """
    Load image, detect map corners, and save cropped version.
    
    Args:
        image_path: Path to input image
        output_path: Path to save cropped image
        padding: Extra pixels to keep around the detected area
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing: {image_name}")
        color = get_dominant_color(image_path)

        # Load image
        with open(image_path, 'rb') as f:
            image_data = np.frombuffer(f.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load {image_path}")
            return False

        # Detect corners
        corners = detect_outer_frame(image, image_name, color)
        if corners is None:
            print(f"{image_name} no outer")
            print("")
            JSON_OUTPUT.append({
                'name': image_name,
                'status': 'OUTER FRAME NOT FOUND'})
            global_failed_colors.append(color)
            failed_processing.append(image_path)
            return 2
        
        x_min, y_min, x_max, y_max = corners
        
        # Apply padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Crop image
        cropped_outer = image[y_min:y_max, x_min:x_max]
        #print(f"Cropped outer {image_path}")
        # Save result
        #cv2.imwrite(OUTPUT_FOLDER + f'\\others\\1outer-{image_name}.jpg', cropped_outer)
        if cropped_outer is None:
            print(f"Cropped outer is none")
            global_failed_colors.append(color)
            return 2
        
        corners_inner = detect_inner_frame(cropped_outer, image_name, color)
        if corners_inner is None:
            print(f"{image_name} no inner")
            print("")
            global_failed_colors.append(color)
            failed_processing.append(image_path)
            cv2.imwrite(OUTPUT_FOLDER + f'\\others\\1outer-{image_name}.jpg', cropped_outer)
            return 3
        elif corners_inner == 1:
            print("Applying simple cut")
            print("")
            x_min, y_min, x_max, y_max = corners
            x_min = x_min + 65
            x_max = x_max - 65
            y_min = y_min + 65
            y_max = y_max - 65

            # Apply padding
            cropped_inner = image[y_min:y_max, x_min:x_max]
            update_output_json('inner', image_name, x_min,y_min,x_max,y_max)
        else:    
            x_min, y_min, x_max, y_max = corners_inner
            # Apply padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(cropped_outer.shape[1], x_max + padding)
            y_max = min(cropped_outer.shape[0], y_max + padding)
            cropped_inner = cropped_outer[y_min:y_max, x_min:x_max]
            update_output_json('inner', image_name, x_min,y_min,x_max,y_max)

        #decode to handle ut8 - åäö
        success, image_encoded = cv2.imencode('.jpg', cropped_inner)
        if success:
            with open(output_path, 'wb') as f:
                f.write(image_encoded.tobytes())
        else:
            print(f"ERROR: Could not encode image")
        #print("")
        #print(f"Successfully cropped: {image_path} → {output_path}")
        return 1
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def batch_process_maps(input_dir: str, output_dir: str, padding: int = 0):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing map images
        output_dir: Directory to save cropped maps
        padding: Extra pixels to keep around the detected area
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = ['*.jpg', '*.png', '*.tif']
    image_files = [f for ext in extensions for f in input_path.glob (f'**/{ext}')]

    print(f"Found {len(image_files)} images to process")
    print("")
    if len(image_files) == 0:
        sys.exit(1)
    
    results = defaultdict(int)
    result_map = {1: 'successful', 2: 'outer', 3: 'inner'}

    for img_file in image_files:
        output_filename = img_file.stem + "-cut.jpg"
        output_file = output_path / output_filename
        JSON_OUTPUT.append({'name':img_file.name, 'status': 'unknown', 'data':{}})
        result = crop_map_image(str(img_file), img_file.name, output_file, padding)
        
        if result in result_map:
            results[result_map[result]] += 1
    
    print("")
    print(f"Successfully processed {results['successful']}/{len(image_files)} images")
    print(f"Outer failed {results['outer']}/{len(image_files)} images")
    print(f"Inner failed {results['inner']}/{len(image_files)} images")
    print("")
    print(f"Colors:")
    for item in list(set(global_colors)):
        print(f"{item}: {global_colors.count(item)}, {global_failed_colors.count(item)} failed")
    print("")
    for item in failed_processing:
        print(f"{item}")
    return

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Usage
if __name__ == "__main__":
    # Single image
    #crop_map_image("map.jpg", "map_cropped.jpg", output_filepath, padding=5)
    
    # Batch processing
    #batch_process_maps("./input_maps", "./output_maps", padding=5)
    #list("")
    if len(sys.argv) < 3:
        print("Usage: python program input-folder output-folder")
        sys.exit(1)
    batch_process_maps(INPUT_FOLDER, OUTPUT_FOLDER, 0)
    #print("writing json")
    with open(OUTPUT_FOLDER+'\\info.json', 'w', encoding='utf-8') as f:
        json.dump(JSON_OUTPUT, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    sys.exit(0)
