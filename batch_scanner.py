import cv2
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from PIL import Image
from ultralytics import YOLO

BOX_MATCH_THRESHOLD = 0.45
CONF_BOX_DETECT = 0.20
CONF_TOY_DETECT = 0.10 
CONF_FLOOR_FILTER = 0.18 # ✨ משתנה חדש: סף מחמיר יותר שמסנן את רעשי הרצפה

toy_model_path = r"C:\Users\win11\Downloads\MODEL\toy_model.pt"
box_model_path = r"C:\Users\win11\Downloads\MODEL\box_model.pt"


#הקוד מטעין את מודלי הYOLO
try:
    toy_model = YOLO(toy_model_path)
    box_model = YOLO(box_model_path)
    print("✅ Batch Scanner YOLO Models loaded!")
except Exception as e:
    print(f"❌ Error loading YOLO models: {e}")
    toy_model = None
    box_model = None

base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#מחלצת את המאפינים
feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
feature_extractor.eval()

#כדי לישתמש בRESNET18 
#צרייך לבעיר את התמונה דרך 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),#מכוצת לריבוע
    transforms.ToTensor(),#הופכת למערך תנזור כדי שהפקסלים שבין 0-255 ימרו ל0.0-1.0
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#מייצבת את המבעים ע"י ממוצע וסטיית תקן
])

#צבעים לקופסאות
def generate_dynamic_color(box_index):
    random.seed(box_index)
    r = random.randint(60, 255)
    g = random.randint(60, 255)
    b = random.randint(60, 255)
    return (b, g, r)


#אוספת לי את כל הנתונים ומחזירה את הוקטור החד ממדי
def extract_vector(cv_img):
    try:
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            vec = feature_extractor(tensor)
        return vec.flatten().unsqueeze(0)
    except:
        return None


##פונקציה שבודקת לי האם קופסא אחת זזה לי והיא אותה קופסא 
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def process_single_image(image_path, save_path, known_boxes_from_db=None):
    img = cv2.imread(image_path)
    if img is None: 
        return None, -1, None, None, None, []

    detected_boxes = []

    # שלב 1: זיהוי קופסאות
    if box_model is not None:
        boxes_out = box_model.predict(img, conf=CONF_BOX_DETECT, verbose=False)[0]
        
        if boxes_out.boxes is not None:
            for i, box in enumerate(boxes_out.boxes):
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                box_id = int(box.id[0]) if (hasattr(box, 'id') and box.id is not None) else (i + 1)
                color = generate_dynamic_color(box_id)
                
                new_box = {
                    'id': box_id,
                    'coords': (bx1, by1, bx2, by2), 
                    'color': color, 
                    'vec': None, 
                    'has_toys': False 
                }

                if known_boxes_from_db:
                    for old_box in known_boxes_from_db:
                        if calculate_iou(new_box['coords'], old_box['coords']) > 0.4:
                            new_box['vec'] = old_box.get('vec')
                            new_box['id'] = old_box['id']
                            new_box['color'] = old_box['color']
                            break

                detected_boxes.append(new_box)

    #  יצירת "זיכרון אקטיבי" לקופסאות
    active_boxes_dict = {b['id']: b for b in known_boxes_from_db} if known_boxes_from_db else {}
    for b in detected_boxes:
        active_boxes_dict[b['id']] = b 
    
    active_boxes = list(active_boxes_dict.values())

    # שלב 2: בדיקת תכולת קופסאות (סינון צעצועים שבפנים)
    toys_out = toy_model.predict(img, conf=CONF_TOY_DETECT, verbose=False)[0]
    floor_toys = [] 

    if toys_out.boxes is not None:
        for toy in toys_out.boxes:
            tx1, ty1, tx2, ty2 = map(int, toy.xyxy[0])
            conf = float(toy.conf[0]) # מחלצים את רמת הביטחון של הזיהוי
            t_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
            is_inside = False
            
            # עכשיו בודקים את המיקום מול הזיכרון האקטיבי
            for b in active_boxes:
                bx1, by1, bx2, by2 = b['coords']
                if bx1 <= t_center[0] <= bx2 and by1 <= t_center[1] <= by2:
                    is_inside = True
                    b['has_toys'] = True 
                    
                    if b['vec'] is None:
                        crop = img[ty1:ty2, tx1:tx2]
                        if crop.size > 0: 
                            b['vec'] = extract_vector(crop)
                    
                    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (150, 150, 150), 1)
                    break
            
            if not is_inside:
                # ✨ הפתרון: צעצוע יירשם על הרצפה רק אם הוא עבר את הסף המחמיר!
                if conf >= CONF_FLOOR_FILTER:
                    floor_toys.append((tx1, ty1, tx2, ty2, t_center))

    # ==========================================
    # ציור הקופסאות על המסך
    # ==========================================
    for b in active_boxes:
        bx1, by1, bx2, by2 = b['coords']
        cv2.rectangle(img, (bx1, by1), (bx2, by2), b['color'], 4)
        if not b['has_toys'] and b.get('vec') is None:
            cv2.rectangle(img, (bx1, by1 - 40), (bx1 + 250, by1), b['color'], -1)
            cv2.putText(img, "EMPTY BOX", (bx1 + 5, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.putText(img, f"Box {b['id']}", (bx1, by1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, b['color'], 3)

    # ==========================================
    # שלב 3: סריקת צעצועי רצפה מהפינה הימנית-תחתונה
    # ==========================================
    floor_toys.sort(key=lambda t: (-t[4][1], -t[4][0])) 

    target_box_id = -1
    target_coords = None
    target_box_coords = None 
    target_color = (180, 180, 180) 
    matched_any = False

    if len(floor_toys) > 0:
        # עוברים על כל הצעצועים כדי למצוא התאמה
        for toy_candidate in floor_toys:
            main_tx1, main_ty1, main_tx2, main_ty2, main_center = toy_candidate
            crop = img[main_ty1:main_ty2, main_tx1:main_tx2]
            vec = extract_vector(crop) 
            
            if vec is not None:
                best_score = -1.0
                best_box = None
                
                # משווים מול הזיכרון האקטיבי
                for b in active_boxes:
                    if b['vec'] is not None:
                        sim = F.cosine_similarity(vec, b['vec']).item()
                        if sim > best_score:
                            best_score = sim
                            best_box = b

                if best_score >= BOX_MATCH_THRESHOLD:
                    target_coords = [main_tx1, main_ty1, main_tx2, main_ty2]
                    target_box_id = best_box['id']
                    target_box_coords = best_box['coords']
                    target_color = best_box['color']
                    matched_any = True
                    thickness = 4 
                    cv2.rectangle(img, (main_tx1, main_ty1), (main_tx2, main_ty2), target_color, thickness)
                    label = f"Match: {best_score:.2f}"
                    cv2.putText(img, label, (main_tx1, main_ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)
                    break 

        # אם עברנו על כולם ואין לאף אחד התאמה
        if not matched_any:
            main_tx1, main_ty1, main_tx2, main_ty2, main_center = floor_toys[0]
            target_coords = [main_tx1, main_ty1, main_tx2, main_ty2]
            target_box_id = 0 
            cv2.rectangle(img, (main_tx1, main_ty1), (main_tx2, main_ty2), target_color, 2)
            cv2.putText(img, "No Match", (main_tx1, main_ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)

    cv2.imwrite(save_path, img)
    # מחזיר לאנגולר ול-API את active_boxes
    return img, target_box_id, target_coords, target_box_coords, target_color, active_boxes