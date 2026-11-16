import os
import shutil
import random

# --- הגדרות נתיבים (תוקן ל"פרויקטם"!) ---
# התיקייה שבה נמצאות התמונות שחתכנו מהסרטונים
src_images = r"C:\טוהר בלולו\פרויקטם\סירטוני גן"
# התיקייה שבה שמרת את קובצי הטקסט (הלייבלים) שחילצת מה-ZIP של MakeSense
src_labels = r"C:\טוהר בלולו\פרויקטם\labels_my-project-name_2026-03-13-12-32-53new" # <--- שמי לב! תשני לנתיב שבו שמת את הלייבלים החדשים

# תיקיית היעד של הפרויקט
dataset_path = r"C:\טוהר בלולו\פרויקטם\drone_project\dataset"

def organize_dataset(split_ratio=0.8):
    # יצירת רשימה של כל התמונות שיש להן תיוג (לייבל)
    all_images = [f for f in os.listdir(src_images) if f.endswith('.jpg')]
    valid_pairs = []

    for img in all_images:
        label_file = img.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(src_labels, label_file)):
            valid_pairs.append(img)

    # ערבוב אקראי
    random.shuffle(valid_pairs)
    
    # חישוב נקודת הפיצול (80-20)
    split_idx = int(len(valid_pairs) * split_ratio)
    train_list = valid_pairs[:split_idx]
    val_list = valid_pairs[split_idx:]

    def move_files(file_list, target_type):
        for img_name in file_list:
            # העברת תמונה
            shutil.copy(os.path.join(src_images, img_name), 
                        os.path.join(dataset_path, "images", target_type, img_name))
            # העברת לייבל
            label_name = img_name.replace('.jpg', '.txt')
            shutil.copy(os.path.join(src_labels, label_name), 
                        os.path.join(dataset_path, "labels", target_type, label_name))

    print(f"מעביר {len(train_list)} תמונות ל-Train...")
    move_files(train_list, "train")
    
    print(f"מעביר {len(val_list)} תמונות ל-Val (מבחן)...")
    move_files(val_list, "val")
    
    print("הכל מוכן! ה-Dataset מאורגן ומוכן לאימון. 🎉")

if __name__ == "__main__":
    organize_dataset()