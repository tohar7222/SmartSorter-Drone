import os
import shutil
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "toys_dataset_v2")

images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
unused_dir = os.path.join(base_dir, "unused_images")

train_images = os.path.join(base_dir, "train", "images")
train_labels = os.path.join(base_dir, "train", "labels")
val_images = os.path.join(base_dir, "val", "images")
val_labels = os.path.join(base_dir, "val", "labels")

for folder in [train_images, train_labels, val_images, val_labels, unused_dir]:
    os.makedirs(folder, exist_ok=True)

print("1. בודק התאמה בין תמונות לתיוגים...")
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))]
valid_images = []

for img_name in all_images:
    txt_name = img_name.rsplit('.', 1)[0] + ".txt"
    if os.path.exists(os.path.join(labels_dir, txt_name)):
        valid_images.append(img_name)
    else:
        shutil.move(os.path.join(images_dir, img_name), os.path.join(unused_dir, img_name))

print(f" {len(valid_images)} תמונות טובות שמוכנות לחלוקה!")

print("2. מחלק לאימון (80%) ומבחן (20%)")
random.shuffle(valid_images)
split_index = int(len(valid_images) * 0.8)
train_list = valid_images[:split_index]
val_list = valid_images[split_index:]

def move_final_files(file_list, dest_images, dest_labels):
    for img_name in file_list:
        shutil.move(os.path.join(images_dir, img_name), os.path.join(dest_images, img_name))
        txt_name = img_name.rsplit('.', 1)[0] + ".txt"
        shutil.move(os.path.join(labels_dir, txt_name), os.path.join(dest_labels, txt_name))

move_final_files(train_list, train_images, train_labels)
move_final_files(val_list, val_images, val_labels)

print("\n--- סיכום פעולה ---")
print("החלוקה החדשה הושלמה בהצלחה!")
print(f" {len(train_list)} תמונות לאימון (train).")
print(f" {len(val_list)} תמונות למבחן (val).")