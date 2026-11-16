import cv2
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

print("מתחיל טעינת מודלים (זה עשוי לקחת מספר שניות)... 🚀")

# ==========================================
# שלב 1: הגדרת כל הנתיבים למודלים ולתמונה שלך
# ==========================================
test_image_path = r"C:\טוהר בלולו\פרויקטם\drone_project\full_room_test.jpg.jpg"

current_dir = os.path.dirname(os.path.abspath(__file__))
toy_model_path = os.path.join(current_dir, "runs", "detect", "toys_model_final3", "weights", "best.pt")
box_model_path = r"C:\טוהר בלולו\פרויקטם\drone_project\runs\detect\train4\weights\best.pt"

# ==========================================
# שלב 2: העלאת המודלים לזיכרון
# ==========================================
toy_model = YOLO(toy_model_path)
box_model = YOLO(box_model_path)

for i in range(len(box_model.names)):
    box_model.names[i] = 'box'

base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
feature_extractor.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_vector(cv_img):
    try:
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            vec = feature_extractor(tensor)
        return vec.flatten().unsqueeze(0)
    except Exception as e:
        print(f"שגיאה בחילוץ וקטור: {e}")
        return None

# ==========================================
# שלב 3: הפונקציה המרכזית - סריקת התמונה
# ==========================================
def scan_room():
    print(f"פותח את התמונה: {test_image_path}")
    img = cv2.imread(test_image_path)
    
    if img is None:
        print("❌ שגיאה: לא הצלחתי לפתוח את התמונה! ודאי שהנתיב מדויק.")
        return

    dynamic_palette = [
        (255, 0, 0),    # כחול
        (0, 255, 0),    # ירוק
        (0, 0, 255),    # אדום
        (0, 255, 255),  # צהוב
        (255, 0, 255),  # סגול
        (255, 255, 0)   # תכלת
    ]
    
    detected_boxes = []

    # --- מציאת קופסאות (הפעם לא מוציאים וקטור מהקופסא עצמה!) ---
    boxes_out = box_model.predict(img, conf=0.85)[0]
    
    for i, box in enumerate(boxes_out.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = dynamic_palette[i % len(dynamic_palette)] 
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, f"Box {i+1}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        
        # שומרים את הקופסא בזיכרון, אבל שמים "None" במקום של ה-DNA
        detected_boxes.append({'color': color, 'vec': None, 'coords': (x1, y1, x2, y2)})

    # --- מציאת צעצועים (חלוקה לאלו שבתוך הקופסא ולאלו שעל הרצפה) ---
    toys_out = toy_model.predict(img, conf=0.25)[0]
    floor_toys = [] # רשימה שתשמור רק את הצעצועים שעל הרצפה
    
    for toy in toys_out.boxes:
        tx1, ty1, tx2, ty2 = map(int, toy.xyxy[0])
        center_x = (tx1 + tx2) // 2
        center_y = (ty1 + ty2) // 2
        
        is_inside_box = False
        
        for b in detected_boxes:
            bx1, by1, bx2, by2 = b['coords']
            # בודקים אם הצעצוע בתוך הקופסא הזו
            if bx1 <= center_x <= bx2 and by1 <= center_y <= by2:
                is_inside_box = True
                
                # !!! כאן קורה השוד !!!
                # אם מצאנו צעצוע בתוך הקופסא, ולקופסא עדיין אין DNA - הצעצוע הזה יהפוך ל-DNA שלה!
                if b['vec'] is None:
                    crop = img[ty1:ty2, tx1:tx2]
                    if crop.size > 0:
                        b['vec'] = extract_vector(crop)
                
                # מסמנים את הצעצוע שבקופסא באפור חלש כרגיל
                cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (150, 150, 150), 1)
                break
                
        # אם הצעצוע לא היה באף קופסא - הוא על הרצפה. שומרים אותו להמשך.
        if not is_inside_box:
            floor_toys.append((tx1, ty1, tx2, ty2))

    # --- מבחן הדמיון: השוואת צעצועי הרצפה ל-DNA שגנבנו ---
    for (tx1, ty1, tx2, ty2) in floor_toys:
        crop = img[ty1:ty2, tx1:tx2]
        if crop.size == 0: continue
        
        vec = extract_vector(crop)
        best_score = -1.0
        best_color = (255, 255, 255)

        # משווים ל-DNA המדויק של הצעצועים מהקופסאות
        if vec is not None:
            for b in detected_boxes:
                if b['vec'] is not None:
                    sim = F.cosine_similarity(vec, b['vec']).item()
                    if sim > best_score:
                        best_score = sim
                        if sim >= 0.50: # סף דמיון לבדיקה
                            best_color = b['color'] 
        
        thickness = 3 if best_score >= 0.50 else 1
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), best_color, thickness)
        cv2.putText(img, f"Toy {best_score:.2f}", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, best_color, 2)

    # התאמת גודל החלון למסך
    cv2.namedWindow("Drone Smart Scanner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Smart Scanner", 1200, 800)
    cv2.imshow("Drone Smart Scanner", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# הפעלת הקוד
if __name__ == "__main__":
    scan_room()