import cv2
import sys
import os
from ultralytics import YOLO

# 1. טעינת המודל הספציפי של הקופסאות (וודאי שהנתיב נכון!)
box_model_path = r"X:\DroneModels\box_model.pt"
model = YOLO(box_model_path)

def process_only_boxes(input_path, output_path):
    # קריאת התמונה
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not read image")
        return

    # 2. הרצת זיהוי - הורדנו את ה-conf ל-0.2 כדי לראות ה-כ-ל
    results = model.predict(source=img, conf=0.2, verbose=False)[0]

    # 3. ציור הריבועים על התמונה
    for box in results.boxes:
        # חילוץ קואורדינטות
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # חילוץ אחוז הביטחון (Confidence)
        confidence = box.conf[0].item()
        
        # ציור ריבוע ירוק עבה
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # כתיבת הטקסט: "BOX" + אחוז הביטחון
        label = f"BOX: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # שמירת התוצאה
    cv2.imwrite(output_path, img)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        process_only_boxes(sys.argv[1], sys.argv[2])
    else:
        print("Waiting for C# to send image...")