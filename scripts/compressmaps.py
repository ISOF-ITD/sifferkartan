import os, sys
from PIL import Image
from pathlib import Path

INPUT_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
JSON_OUTPUT = [] # append per action taken for each image


def compress_images_batch(input_dir, output_dir, quality=85, max_width=None):
    """Compress all JPGs with progress tracking."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    jpg_files = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.jpeg'))
    
    if not jpg_files:
        print("No JPG files found.")
        return
    
    total_original = 0
    total_compressed = 0
    
    for i, input_path in enumerate(jpg_files, 1):
        try:
            img = Image.open(input_path)
            
            if max_width and img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            output_path = Path(output_dir) / input_path.name
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            orig_size = input_path.stat().st_size / 1024
            comp_size = output_path.stat().st_size / 1024
            total_original += orig_size
            total_compressed += comp_size
            
            print(f"[{i}/{len(jpg_files)}] {input_path.name}: {orig_size:.1f}KB → {comp_size:.1f}KB")
        
        except Exception as e:
            print(f"Error: {input_path.name} - {e}")
    
    overall_reduction = ((total_original - total_compressed) / total_original) * 100
    print(f"\n✓ Complete! Total: {total_original:.1f}KB → {total_compressed:.1f}KB ({overall_reduction:.1f}% reduction)")


# Usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python program input-folder output-folder")
        print("alternate: python program input-folder output-folder quality(1-100)")
        sys.exit(1)
    elif len(sys.argv) > 3:
        print("using specified quality")
        compress_images_batch(INPUT_FOLDER, OUTPUT_FOLDER, sys.argv[3])
    else:
        compress_images_batch(INPUT_FOLDER, OUTPUT_FOLDER, quality=10)

