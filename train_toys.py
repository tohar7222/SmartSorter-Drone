from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. מכוונים בדיוק לקובץ ההצלה שהמחשב שמר
last_weights_path = os.path.join(current_dir, "runs", "detect", "toys_model_final3", "weights", "last.pt")

# 2. טוענים את ה"מוח" מנקודת העצירה במקום להתחיל מאפס
model = YOLO(last_weights_path) 

print("🚀 משחזר נתונים... ממשיך את האימון בדיוק מהסיבוב שעצרנו בו!")

# 3. פקודת הקסם: resume=True (YOLO כבר זוכר לבד שצריך 100 סיבובים ואת כל ההגדרות)
results = model.train(resume=True)

print("האימון הסתיים בהצלחה!")