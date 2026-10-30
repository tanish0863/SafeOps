import os
import shutil

# Paths
dataset_dir = r"d:\Projects\Hackive (angadi)\vision\datasets\with-or-without-helmet.yolov8"
train_images = os.path.join(dataset_dir, "train", "images")
train_labels = os.path.join(dataset_dir, "train", "labels")
dest_helmet = r"d:\Projects\Hackive (angadi)\vision\samples\helmet"
dest_no_helmet = r"d:\Projects\Hackive (angadi)\vision\samples\no_helmet"

# Ensure dest dirs exist
os.makedirs(dest_helmet, exist_ok=True)
os.makedirs(dest_no_helmet, exist_ok=True)

# Classes: 0: bike, 1: with_helmet, 2: without_helmet
class_helmet = 1
class_no_helmet = 2

files = os.listdir(train_labels)
found_helmet = []
found_no_helmet = []

for f in files:
    if not f.endswith('.txt'): continue
    label_path = os.path.join(train_labels, f)
    image_name = f.rsplit('.', 1)[0] + '.jpg'
    image_path = os.path.join(train_images, image_name)
    
    if not os.path.exists(image_path):
        continue
        
    with open(label_path, 'r') as lf:
        lines = lf.readlines()
        classes = set()
        for line in lines:
            parts = line.split()
            if parts:
                classes.add(int(parts[0]))
        
        # Prefer images with ONLY one type for purity, or at least the target type
        if class_helmet in classes and len(found_helmet) < 20:
            found_helmet.append((image_path, image_name))
        
        if class_no_helmet in classes and len(found_no_helmet) < 20:
            found_no_helmet.append((image_path, image_name))
            
    if len(found_helmet) >= 20 and len(found_no_helmet) >= 20:
        break

print(f"Found {len(found_helmet)} helmet images and {len(found_no_helmet)} no-helmet images.")

for src, name in found_helmet:
    shutil.copy(src, os.path.join(dest_helmet, name))
    
for src, name in found_no_helmet:
    shutil.copy(src, os.path.join(dest_no_helmet, name))

print("Done copying.")
