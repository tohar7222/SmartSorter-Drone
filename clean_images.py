import os
from PIL import Image
import shutil

images_dir = 'ALL_IMAGES'
bad_images_dir = 'BAD_IMAGES'

os.makedirs(bad_images_dir, exist_ok=True)

good_count = 0
bad_count = 0

print("מתחיל להפעיל 'מכונת כביסה' על התמונות... נא להמתין.")

for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        filepath = os.path.join(images_dir, filename)
        try:
            with Image.open(filepath) as img:
                img.verify() 
            
            with Image.open(filepath) as img:
                rgb_im = img.convert('RGB')
                rgb_im.save(filepath, 'JPEG')
            
            good_count += 1
            
        except Exception as e:
            bad_filepath = os.path.join(bad_images_dir, filename)
            shutil.move(filepath, bad_filepath)
            bad_count += 1

print("--- סיכום ניקוי ---")
print(f"✅ {good_count} תמונות תוקנו, נוקו, ומוכנות לאימון.")
print(f"❌ {bad_count} תמונות היו פגומות לחלוטין והועברו לתיקיית {bad_images_dir}.")