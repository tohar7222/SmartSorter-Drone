import cv2
import os

videos_folder = r"C:\טוהר בלולו\פרויקטם\סירטוני גן" 

print("מתחיל לעבוד.")

os.chdir(videos_folder)

total_saved = 0

for filename in os.listdir("."):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        print(f"חותך עכשיו את הסרטון: {filename} ")
        
        cap = cv2.VideoCapture(filename)
        
        if not cap.isOpened():
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        if fps <= 0:
            fps = 30 
            
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            
            if frame_count % fps == 0:
                image_name = f"training_frame_{total_saved}.jpg"
                
                cv2.imwrite(image_name, frame)
                total_saved += 1
                
            frame_count += 1
            
        cap.release()

print(f"{total_saved} תמונות ליד הסרטונים.")