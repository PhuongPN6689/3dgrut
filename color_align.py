import os
import sys
import glob
import argparse
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Align exposure and color channels across all images to a reference view.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the training images to be modified in-place")
    return parser.parse_args()

def main():
    args = parse_args()
    image_dir = args.image_dir
    print(f"[*] Running global color alignment on: {image_dir}")
    
    # Find all images
    extensions = ("*.JPG", "*.jpg", "*.png", "*.PNG", "*.jpeg", "*.JPEG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print("[-] Error: No images found in directory.")
        sys.exit(1)
        
    # 1. Select the reference image (using the middle image of the sorted list as typical exposure reference)
    ref_idx = len(image_paths) // 2
    ref_path = image_paths[ref_idx]
    print(f"[+] Using reference image: {ref_path}")
    
    try:
        ref_img = np.array(Image.open(ref_path)).astype(np.float32)
    except Exception as e:
        print(f"[-] Error loading reference image {ref_path}: {e}")
        sys.exit(1)
        
    ref_mean = np.mean(ref_img, axis=(0, 1))
    ref_std = np.std(ref_img, axis=(0, 1))
    ref_std = np.clip(ref_std, 1e-5, None)
    
    # 2. Process all images in-place
    success_count = 0
    for p in image_paths:
        if p == ref_path:
            success_count += 1
            continue
        try:
            img = np.array(Image.open(p)).astype(np.float32)
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
            std = np.clip(std, 1e-5, None)
            
            # Match mean and standard deviation per channel
            matched = (img - mean) * (ref_std / std) + ref_mean
            matched = np.clip(matched, 0, 255).astype(np.uint8)
            
            Image.fromarray(matched).save(p)
            success_count += 1
        except Exception as e:
            print(f"[-] Error processing {p}: {e}")
            
    print(f"[+] Successfully aligned exposure for {success_count}/{len(image_paths)} images.")

if __name__ == "__main__":
    main()
