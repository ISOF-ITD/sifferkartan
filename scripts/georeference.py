from osgeo import gdal, osr
import numpy as np
from pathlib import Path
import json, sys

INPUT_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
JSON_INPUT = sys.argv[3]
JSON_OUTPUT = [] # append per action taken for each image

def main():
    if len(sys.argv) < 3:
        print("Usage: python program input-folder output-folder")
        sys.exit(1)
    batch_process_maps(INPUT_FOLDER, OUTPUT_FOLDER, JSON_INPUT)
    print("writing json")
    with open(OUTPUT_FOLDER+'\\geotiff-info.json', 'w') as f:
        json.dump(JSON_OUTPUT, f, indent=4)


def batch_process_maps(input_folder: str, output_folder: str, json_path: str, log_level: int = 0):
    print(input_folder)
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process all common image formats
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + \
                  list(input_path.glob('*.tif'))
    
    print(f"Found {len(image_files)} images to process")
    if len(image_files) == 0:
        sys.exit(1)

    successful = 0
    # Load geospatial data
    with open(json_path, 'r') as f:
        geo_info = json.load(f)

    # get the json info with the Feature -> kartbladsid = img_file.name
    for img_file in image_files:
        output_filename = img_file.name.rsplit(".")[0] + ".tiff"
        output_file = output_path / output_filename
        id = img_file.name.rsplit("_")[1].rsplit(".")[0]
        print(id)
        if process_tif(str(img_file), geo_info, id, str(output_file)):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(image_files)} images")

def process_tif(tif_path, geo_info, kartbladsid, output_path):
    """
    Create a GeoTIFF from a TIF file using geospatial data from GeoJSON.
    
    Args:
        tif_path: Path to the input TIF file
        geo_info: The loaded GeoJSON data (FeatureCollection)
        kartbladsid: The kartbladsid to look up in the GeoJSON
        output_path: Path for the output GeoTIFF
    """
    
    # Find the feature matching the kartbladsid
    coords = None
    for feat in geo_info['features']:
        if feat['properties']['kartbladsid'] == kartbladsid:
            coords = feat['geometry']['coordinates'][0]
            break
    
    if coords is None:
        raise ValueError(f"kartbladsid '{kartbladsid}' not found in geo_info")
    
    # Open source TIF
    source = gdal.Open(tif_path)
    if source is None:
        raise FileNotFoundError(f"Cannot open {tif_path}")
    
    # Get raster dimensions
    width = source.RasterXSize
    height = source.RasterYSize
    
    # Calculate geotransform from polygon bounds
    # Extract min/max coordinates
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    
    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)
    
    # Calculate pixel sizes
    pixel_width = (max_lon - min_lon) / width
    pixel_height = (min_lat - max_lat) / height  # Negative because y increases downward
    
    # Create geotransform (upper_left_x, pixel_width, 0, upper_left_y, 0, pixel_height)
    geotransform = (min_lon, pixel_width, 0, max_lat, 0, pixel_height)
    
    # Create GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    geotiff = driver.CreateCopy(output_path, source)
    
    # Apply geotransform
    geotiff.SetGeoTransform(geotransform)
    
    # Set CRS (EPSG:4326 = WGS84, which your GeoJSON uses)
    # 4979 is CRS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4979)
    geotiff.SetProjection(srs.ExportToWkt())
    
    # Flush to disk
    geotiff.FlushCache()
    geotiff = None
    source = None
    
    print(f"GeoTIFF created: {output_path}")
    return True
    #sys.exit(1)


main()