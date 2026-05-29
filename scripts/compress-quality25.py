import sys
from pathlib import Path
from osgeo import gdal

def compress_tif_to_jpeg(input_file, output_file, jpeg_quality=25):
    """
    Compress a GeoTIFF file using JPEG compression via GDAL Python library.
    Separates RGB bands from alpha, applies YCBCR photometric interpretation,
    and creates an internal mask for proper transparency in GeoServer.
    
    Args:
        input_file (str): Path to input GeoTIFF file
        output_file (str): Path to output compressed GeoTIFF file
        jpeg_quality (int): JPEG quality (1-100, default 25)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the input dataset
        source_ds = gdal.Open(input_file)
        if source_ds is None:
            print(f"✗ Failed to open {input_file}")
            return False
        
        # Get band count
        band_count = source_ds.RasterCount
        
        # Determine which bands to use
        if band_count >= 4:
            # Has alpha band: use RGB (1-3) + alpha mask (4)
            bands = [1, 2, 3]
            mask_band = 4
        elif band_count >= 3:
            # RGB only, no alpha
            bands = [1, 2, 3]
            mask_band = None
        else:
            # Grayscale or single band
            bands = list(range(1, band_count + 1))
            mask_band = None
        
        # Create translate options
        creation_options = [
            'COMPRESS=JPEG',
            f'JPEG_QUALITY={jpeg_quality}',
            'TILED=YES',
            'PHOTOMETRIC=YCBCR'  # Now safe to use with 3-band RGB
        ]
        
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            bandList=bands,
            maskBand=mask_band if mask_band else None,
            creationOptions=creation_options
        )
        
        # Enable internal GDAL TIFF mask
        gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
        
        # Translate (compress) the file
        output_ds = gdal.Translate(output_file, source_ds, options=translate_options)
        
        if output_ds is None:
            print(f"✗ Failed to compress {input_file}")
            return False
        
        # If mask band was specified, create the internal mask
        if mask_band:
            output_band = output_ds.GetRasterBand(1)
            if output_band:
                output_band.CreateMask(gdal.GMF_PER_DATASET)
        
        # Close datasets
        output_ds = None
        source_ds = None
        
        return True
    except Exception as e:
        print(f"✗ Error compressing {input_file}: {str(e)}")
        return False

def batch_compress_directory(input_dir, output_dir, jpeg_quality=25):
    """
    Compress all GeoTIFF files in a directory recursively.
    
    Args:
        input_dir (str): Directory containing input GeoTIFF files
        output_dir (str): Directory to save compressed files
        jpeg_quality (int): JPEG quality (1-100, default 25)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .tif files recursively
    tif_files = list(input_path.rglob("*.tif")) + list(input_path.rglob("*.TIF"))
    
    if not tif_files:
        print(f"No .tif files found in '{input_dir}'")
        sys.exit(1)
    
    print(f"Found {len(tif_files)} .tif file(s) to compress\n")
    
    success = 0
    failed = 0
    total = len(tif_files)
    
    for index, input_file in enumerate(tif_files, 1):
        # Preserve directory structure in output folder
        relative_path = input_file.relative_to(input_path)
        output_file = output_path / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        progress = f"[{index}/{total}]"
        print(f"{progress} ", end="", flush=True)
        
        if compress_tif_to_jpeg(str(input_file), str(output_file), jpeg_quality):
            print(f"✓ {input_file}")
            success += 1
        else:
            print(f"✗ {input_file}")
            failed += 1
    
    print(f"\nBatch operation complete: {success}/{total} files compressed successfully")
    if failed > 0:
        print(f"Failed: {failed} file(s)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_folder> <output_folder> [jpeg_quality]")
        print("Example: python script.py ./input_data ./output_data 25")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    
    batch_compress_directory(input_folder, output_folder, quality)
