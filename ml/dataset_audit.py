import os
import random
from PIL import Image
import collections

# CONFIGURATION
DATASET_PATH = r"C:\Users\Yash\OneDrive\Desktop\DIPEX\PlantVillage"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def perform_audit():
    print("="*60)
    print("      PLANT DISEASE DATASET QUALITY AUDIT REPORT")
    print("="*60)

    # 1. Structure Verification
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset path not found: {DATASET_PATH}")
        return

    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    classes.sort()
    
    print(f"\n[1] Structure Verification:")
    print(f"    - Subfolders found (Classes): {len(classes)}")
    print(f"    - First 10 class names:")
    for c in classes[:10]:
        print(f"      • {c}")

    # 2. Content Validity & Class Balance
    total_images = 0
    class_counts = {}
    valid_images_list = []

    for c in classes:
        class_path = os.path.join(DATASET_PATH, c)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
        count = len(images)
        class_counts[c] = count
        total_images += count
        for img in images:
            valid_images_list.append(os.path.join(class_path, img))

    print(f"\n[2] Content Validity:")
    print(f"    - Total valid images (JPG/PNG/JPEG): {total_images}")
    print(f"    - Total classes: {len(classes)}")

    # 3. Class Balance Check
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n[3] Class Balance Check:")
    print(f"    - Top 5 classes (Most images):")
    for name, count in sorted_classes[:5]:
        print(f"      • {name}: {count} images")
    
    print(f"    - Bottom 5 classes (Fewest images):")
    for name, count in sorted_classes[-5:]:
        print(f"      • {name}: {count} images")

    low_count_classes = [name for name, count in class_counts.items() if count < 100]
    if low_count_classes:
        print(f"\n    ⚠️  WARNING: Found {len(low_count_classes)} classes with fewer than 100 images:")
        for name in low_count_classes:
            print(f"      - {name} ({class_counts[name]} images)")
    else:
        print(f"\n    ✅ All classes have at least 100 images.")

    # 4. Image Quality Spot-Check
    print(f"\n[4] Image Quality Spot-Check:")
    if valid_images_list:
        random_img_path = random.choice(valid_images_list)
        try:
            with Image.open(random_img_path) as img:
                width, height = img.size
                mode = img.mode
                print(f"    - Extracted random image: {os.path.basename(random_img_path)}")
                print(f"    - Dimensions: {width}x{height}")
                print(f"    - Color Mode: {mode}")
                print(f"    - Readable: YES")
        except Exception as e:
            print(f"    - Readable: NO (Error: {str(e)})")
    else:
        print(f"    - No images found to check.")

    # 5. Final Verdict
    print(f"\n[5] Final Verdict:")
    
    # Heuristic for agricultural data based on keywords
    agri_keywords = ['leaf', 'spot', 'rot', 'blight', 'healthy', 'mildew', 'virus', 'rust', 'scab', 'tomato', 'potato', 'corn', 'grape', 'apple', 'pepper']
    found_keywords = any(any(kw in c.lower() for kw in agri_keywords) for c in classes)
    
    looks_like_agri = "YES" if found_keywords else "NO"
    ready_for_transfer = "YES" if (len(classes) > 1 and total_images > 1000 and not any(c == 0 for c in class_counts.values())) else "NO"

    print(f"    - Does this look like agricultural/plant disease data? {looks_like_agri}")
    print(f"    - Is this dataset ready for Transfer Learning? {ready_for_transfer}")
    
    if ready_for_transfer == "YES":
        print("\n✅ VERDICT: PRE-TRAINING READY")
    else:
        print("\n❌ VERDICT: NOT READY - Check error/warning logs.")

    print("\n" + "="*60)

if __name__ == "__main__":
    perform_audit()
