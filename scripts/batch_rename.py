from pathlib import Path
import sys

"""
Way to change name of multiple files at once.

Usage:
    python3 batch_rename folder-in folder-out
"""

def main():
    if len(sys.argv) < 3:
        print("Usage: python program input-folder output-folder")
        sys.exit(1)
    batch_rename(sys.argv[1], sys.argv[2])

def batch_rename(
    input_folder: str, output_folder: str
):
    print(input_folder)
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ["*.jpg", "*.png", "*.tiff"]
    image_files = [f for ext in extensions for f in input_path.glob(f"**/{ext}")]

    print(f"Found {len(image_files)} images to process")
    if len(image_files) == 0:
        sys.exit(1)

    successful = 0
    failed_data = 0
    failed_other = 0

    # get the json info with the Feature -> kartbladsid = img_file.name
    for i, img_file in enumerate(image_files, 1):
        subdir_path = Path(output_folder) / img_file.parent.stem
        subdir_path.mkdir(parents=True, exist_ok=True)
        output_filename = img_file.stem + "_modified_auto.tiff"
        output_file = subdir_path / output_filename
        print(
            f"[{failed_data}][{failed_other}][{i}/{len(image_files)}] Processing: {img_file.name}"
        )
        try:
            img_file.rename(output_file)
            result = True
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            result = 1
        
        match result:
            case True:
                successful += 1
            case False:
                failed_data += 1
            case 1:
                failed_other += 1

    print("")
    print(f"Successfully processed {successful}/{len(image_files)} images")


main()