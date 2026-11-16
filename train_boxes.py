from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_path = os.path.join(current_dir, "data.yaml")

# 1. טעינת מודל YOLOv8 בסיסי
model = YOLO("yolov8n.pt") 

print(" מתחיל אימון מודל לזיהוי קופסאות (כללי, פתוחות וסגורות)...")

results = model.train(
    data=data_yaml_path,       
    epochs=100,                
    imgsz=640,                  
    name="boxes_3_classes_model" 
)

print("אימון מודל הקופסאות המשולב הסתיים בהצלחה!")