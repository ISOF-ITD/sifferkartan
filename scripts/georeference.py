from osgeo import gdal, osr
import numpy as np
from pathlib import Path
import json, sys

INPUT_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
JSON_GEOREFERENCE = sys.argv[3]
JSON_OUTPUT = [] # append per action taken for each image
JSON_MAP_INFO = ""
RUN_MODE = 0

def main():
    if len(sys.argv) < 4:
        print("Usage: python program input-folder output-folder json-input")
        sys.exit(1)
    if len(sys.argv) > 4:
        global RUN_MODE
        RUN_MODE = 1
        global JSON_MAP_INFO
        JSON_MAP_INFO= sys.argv[4]
    batch_process_maps(INPUT_FOLDER, OUTPUT_FOLDER, JSON_GEOREFERENCE)
    print(sys.argv)
    #with open(OUTPUT_FOLDER+'\\geotiff-info.json', 'w') as f:
    #    json.dump(JSON_OUTPUT, f, indent=4)


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
    if RUN_MODE == 1:
        with open(JSON_MAP_INFO, 'r', encoding='utf-8') as b:
            map_info = json.load(b)
    

    # get the json info with the Feature -> kartbladsid = img_file.name
    for img_file in image_files:
        output_filename = img_file.name.rsplit(".")[0] + ".tiff"
        output_file = output_path / output_filename
        if RUN_MODE == 0:
            id = img_file.name.rsplit("_")[1].rsplit("-cut")[0]
            if process_tif(str(img_file), geo_info, id, str(output_file)):
                successful += 1
        elif RUN_MODE == 1:
            id = img_file.name.rsplit("_")[1]
            if process_original(str(img_file), geo_info, map_info, id, str(output_file)):
                successful += 1
    
    print(f"Successfully processed {successful}/{len(image_files)} images")

def process_original(tif_path, geo_info, map_info, kartbladsid, output_path):
    '''
    Processes original unprocessed map-files, jpgs.
    Puts coordinates in the map corners of the original images.
    
    Use with json-output from cutmaps.py, which handles corner detection. 
    '''

    # TODO:
    # FIX ROTATION, RIGHT NOW ITS A BIT OFF. CHECK OUTPUT, FIX.
    
    coords_geo_info = None
    print(kartbladsid)
    just_kartbladsid = kartbladsid.rsplit(' ')[0]
    for feat in geo_info['features']:
        if feat['properties']['kartbladsid'] == just_kartbladsid:
            coords_geo_info = feat['geometry']['coordinates'][0]
            break

    if coords_geo_info is None:
        print(f"kartbladsid '{kartbladsid}' not found in geo_info")
        return False
        
    coords_map_info = []
    for feat in map_info:
        if just_kartbladsid in feat['name']:
            coords_map_info.append(feat['data'])
    
    coords_map_set = []
    if coords_map_info is None or 'inner' not in str(coords_map_info):
        print(f"kartbladsid '{kartbladsid}' does not have needed data")
        return False
    else:
        # Math to get corners of the map.
        coords_map_set = coords_map_info[0]['outer']['coords'] 
        coords_map_set += coords_map_info[0]['inner']['coords'] 
        coords_map_set = [ coords_map_set[0] + coords_map_set[4],
                          coords_map_set[1] + coords_map_set[5],
                          coords_map_set[2] - (coords_map_set[6] - coords_map_set[2]),
                          coords_map_set[3] - (coords_map_set[7] - coords_map_set[3])] 
    
    # Open source TIF
    source = gdal.Open(tif_path)
    if source is None:
        raise FileNotFoundError(f"Cannot open {tif_path}")
    
    # Get the four corners of the polygon (remove duplicate last point)
    polygon_corners = coords_geo_info[:-1]
    
    if len(polygon_corners) != 4:
        raise ValueError(f"Expected 4 corners, found {len(polygon_corners)}. "
                        f"Polygon must be a quadrilateral for proper georeference.")
    
    # Create GCPs by mapping polygon corners directly to image corners
    # GeoJSON polygon corners are typically ordered: top-left, top-right, bottom-right, bottom-left
    # Map them to image corners in the same order
    gcps = []
    image_corners = [
        (coords_map_set[0], coords_map_set[3]),                 
        (coords_map_set[0], coords_map_set[1]),                  
        (coords_map_set[2], coords_map_set[1]),                  
        (coords_map_set[2], coords_map_set[3]),                  
    ]

    for i, (poly_corner, img_corner) in enumerate(zip(polygon_corners, image_corners)):
        gcp = gdal.GCP(poly_corner[0], poly_corner[1], 0, img_corner[0], img_corner[1])
        gcps.append(gcp)
    
    # Create temporary GeoTIFF with GCPs
    temp_output = output_path.replace(".tiff", "_temp.tiff")
    driver = gdal.GetDriverByName('GTiff')
    geotiff = driver.CreateCopy(temp_output, source)
    
    # Set GCPs and CRS
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4979)
    geotiff.SetGCPs(gcps, srs.ExportToWkt())
    
    geotiff.FlushCache()
    geotiff = None
    source = None
    
    # Reproject from EPSG:4979 to EPSG:3006
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4979)
    
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(3006)
    
    warp_options = gdal.WarpOptions(
        srcSRS=src_srs,
        dstSRS=dst_srs,
        resampleAlg=gdal.GRA_Bilinear
    )
    
    gdal.Warp(output_path, temp_output, options=warp_options)
    
    # Remove temporary file
    import os
    os.remove(temp_output)
    
    print(f"GeoTIFF created and reprojected: {output_path}")
    return True

    sys.exit(0)
    sys.exit(0)
    sys.exit(0)
    sys.exit(0)
    sys.exit(0)
    return True


def process_tif(tif_path, geo_info, kartbladsid, output_path):
    """
    Create a GeoTIFF from a TIF file using geospatial data from GeoJSON,
    then reproject from EPSG:4979 to EPSG:3006.
    
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
    
    # Get the four corners of the polygon (remove duplicate last point)
    polygon_corners = coords[:-1]
    
    if len(polygon_corners) != 4:
        raise ValueError(f"Expected 4 corners, found {len(polygon_corners)}. "
                        f"Polygon must be a quadrilateral for proper georeference.")
    
    # Create GCPs by mapping polygon corners directly to image corners
    # GeoJSON polygon corners are typically ordered: top-left, top-right, bottom-right, bottom-left
    # Map them to image corners in the same order
    gcps = []
    image_corners = [
        (0, height),                # bottom-left
        (0, 0),                    # top-left
        (width, 0),                # top-right
        (width, height),           # bottom-right
    ]
    
    for i, (poly_corner, img_corner) in enumerate(zip(polygon_corners, image_corners)):
        gcp = gdal.GCP(poly_corner[0], poly_corner[1], 0, img_corner[0], img_corner[1])
        gcps.append(gcp)
    
    # Create temporary GeoTIFF with GCPs
    temp_output = output_path.replace(".tiff", "_temp.tiff")
    driver = gdal.GetDriverByName('GTiff')
    geotiff = driver.CreateCopy(temp_output, source)
    
    # Set GCPs and CRS
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4979)
    geotiff.SetGCPs(gcps, srs.ExportToWkt())
    
    geotiff.FlushCache()
    geotiff = None
    source = None
    
    # Reproject from EPSG:4979 to EPSG:3006
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4979)
    
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(3006)
    
    warp_options = gdal.WarpOptions(
        srcSRS=src_srs,
        dstSRS=dst_srs,
        resampleAlg=gdal.GRA_Bilinear
    )
    
    gdal.Warp(output_path, temp_output, options=warp_options)
    
    # Remove temporary file
    import os
    os.remove(temp_output)
    
    print(f"GeoTIFF created and reprojected: {output_path}")
    return True

    #sys.exit(1)


main()