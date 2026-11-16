from ultralytics import YOLO

# 1. טעינת המודל המנצח שלך
model_path = r"C:\טוהר בלולו\פרויקטם\drone_project\runs\detect\train4\weights\best.pt"
model = YOLO(model_path)

# --- הקסם: דורסים את השמות של המחלקות! ---
# במקום שמות שונים, נגיד לו לקרוא לכולם פשוט "box"
model.names[0] = 'box'
model.names[1] = 'box'
model.names[2] = 'box'

# 2. נתיב לתמונה שלך 
image_path = r"C:\Users\win11\Downloads\mjoelkkanna-basket-bamboo__1224572_pe915039_s5.avif" 

# 3. הפעלת הזיהוי
results = model.predict(source=image_path, save=True, conf=0.25)

print("הבדיקה הסתיימה! התמונה עם השם האחיד נשמרה בתיקיית runs/detect/predict החדשה")