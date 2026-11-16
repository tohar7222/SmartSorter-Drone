import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from PIL import Image
from ultralytics import YOLO

# ==========================================
# ייבוא ספריית השליטה ברחפן הפיזי
# ==========================================
from djitellopy import Tello
import time

class DroneVisionSystem:
    def __init__(self):
        print("🚁 Drone Vision System Initializing...")
        self.CONF_BOX_DETECT = 0.25
        self.CONF_TOY_DETECT = 0.10
        self.BOX_MATCH_THRESHOLD = 0.45

        self.toy_model = YOLO(r"C:\Users\win11\Downloads\MODEL\toy_model.pt")
        self.box_model = YOLO(r"C:\Users\win11\Downloads\MODEL\box_model.pt")

        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 

    def generate_dynamic_color(self, box_index):
        random.seed(box_index)
        r = random.randint(60, 255)
        g = random.randint(60, 255)
        b = random.randint(60, 255)
        return (b, g, r)

    def extract_vector(self, cv_img):
        try:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tensor = self.preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                vec = self.feature_extractor(tensor)
            return vec.flatten().unsqueeze(0)
        except:
            return None

#זיהוי הקופסאות ושימרה שלהן 
    def phase1_map_boxes(self, high_altitude_img):
        mapped_boxes = []
        boxes_out = self.box_model.predict(high_altitude_img, conf=self.CONF_BOX_DETECT, verbose=False)[0]
        toys_out = self.toy_model.predict(high_altitude_img, conf=self.CONF_TOY_DETECT, verbose=False)[0]
        
        if boxes_out.boxes is None:
            return mapped_boxes

        for i, box in enumerate(boxes_out.boxes):
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            box_id = int(box.id[0]) if (hasattr(box, 'id') and box.id is not None) else (i + 1)
            color = self.generate_dynamic_color(box_id)
            
            box_data = {
                'id': box_id,
                'coords': (bx1, by1, bx2, by2),
                'color': color,
                'vec': None
            }

            if toys_out.boxes is not None:
                for toy in toys_out.boxes:
                    tx1, ty1, tx2, ty2 = map(int, toy.xyxy[0])
                    t_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                    
                    if bx1 <= t_center[0] <= bx2 and by1 <= t_center[1] <= by2:
                        crop = high_altitude_img[ty1:ty2, tx1:tx2]
                        if crop.size > 0:
                            box_data['vec'] = self.extract_vector(crop)
                        break 
            
            mapped_boxes.append(box_data)
            
        return mapped_boxes

    def phase2_sweep_toys(self, sweep_img, known_boxes_from_db):
        toys_out = self.toy_model.predict(sweep_img, conf=self.CONF_TOY_DETECT, verbose=False)[0]
        floor_toys = []

        if toys_out.boxes is not None:
            for toy in toys_out.boxes:
                tx1, ty1, tx2, ty2 = map(int, toy.xyxy[0])
                t_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                
                is_in_box = False
                for b in known_boxes_from_db:
                    bx1, by1, bx2, by2 = b['coords']
                    if bx1 <= t_center[0] <= bx2 and by1 <= t_center[1] <= by2:
                        is_in_box = True
                        break
                
                if not is_in_box:
                    floor_toys.append((tx1, ty1, tx2, ty2, t_center))

        floor_toys.sort(key=lambda t: (-t[4][1], -t[4][0]))

        target_box_id = -1
        target_coords = None

        if len(floor_toys) > 0:
            main_tx1, main_ty1, main_tx2, main_ty2, main_center = floor_toys[0]
            target_coords = [main_tx1, main_ty1, main_tx2, main_ty2]
            target_box_id = 0

        for (tx1, ty1, tx2, ty2, t_center) in floor_toys:
            crop = sweep_img[ty1:ty2, tx1:tx2]
            vec = self.extract_vector(crop)
            if vec is None: continue

            best_score = -1.0
            best_color = (180, 180, 180)
            best_box_id = None

            for b in known_boxes_from_db:
                if b['vec'] is not None:
                    sim = F.cosine_similarity(vec, b['vec']).item()
                    if sim > best_score:
                        best_score = sim
                        if sim >= self.BOX_MATCH_THRESHOLD:
                            best_color = b['color']
                            best_box_id = b['id']
                            
                            if t_center == floor_toys[0][4]:
                                target_box_id = best_box_id

            thickness = 4 if best_score >= self.BOX_MATCH_THRESHOLD else 2
            cv2.rectangle(sweep_img, (tx1, ty1), (tx2, ty2), best_color, thickness)
            
            label = f"Match Box {best_box_id}" if best_box_id is not None else "No Match"
            cv2.putText(sweep_img, label, (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, best_color, 2)

        return sweep_img, target_box_id, target_coords


# ====================================================
# לולאת הטיסה והשליטה ברחפן הפיזי (Main Execution)
# ====================================================
if __name__ == "__main__":
    
    # 1. הפעלת מוח המערכת
    drone_brain = DroneVisionSystem()
    
    # 2. התחברות לרחפן (המחשב חייב להיות מחובר ל-WiFi של הרחפן!)
    tello = Tello()
    tello.connect()
    print(f" Battery: {tello.get_battery()}%")
    
    # 3. הפעלת המצלמה של הרחפן
    tello.streamon()
    time.sleep(2) # ממתינים קצת כדי שהווידאו יתייצב
    
    tello.takeoff() #המראה
    
    
    boxes_in_db = [] 
    
    # מרכז התמונה של Tello בדרך כלל (רוחב 960 פיקסלים)
    SCREEN_CENTER_X = 960 // 2 

    try:
        print("🚁 מתחיל בסריקה חיה...")
        while True:
            # 4. קבלת פריים
            frame = tello.get_frame_read().frame
            
            # 5. עיבוד התמונה: מחפשים צעצועים ומקבלים יעד
            processed_img, target_box_id, target_coords = drone_brain.phase2_sweep_toys(frame, boxes_in_db)
            
            # 6. מנגנון הניווט (Navigation Logic)
            if target_coords is not None:
                # הפירוק של מיקום הצעצוע המטרה
                tx1, ty1, tx2, ty2 = target_coords
                toy_center_x = (tx1 + tx2) // 2
                
                # חישוב הסטייה: כמה הצעצוע רחוק מאמצע המסך?
                error_x = toy_center_x - SCREEN_CENTER_X
                
                # מהירויות טיסה
                speed = 20
                
                # פקודת התנועה עובדת כך: (שמאלה/ימינה, קדימה/אחורה, למעלה/למטה, סיבוב)
                if error_x > 80:
                    print(f"➡️ צעצוע מימין (סטייה {error_x}). טס ימינה...")
                    # tello.send_rc_control(speed, 0, 0, 0)
                elif error_x < -80:
                    print(f"⬅️ צעצוע משמאל (סטייה {error_x}). טס שמאלה...")
                    # tello.send_rc_control(-speed, 0, 0, 0)
                else:
                    print("⬆️ צעצוע ממול ובמרכז! טס קדימה כדי לאסוף...")
                    # tello.send_rc_control(0, speed, 0, 0)
            else:
                print("🔍 מרחף במקום, מחפש צעצועים...")
                # tello.send_rc_control(0, 0, 0, 0)
                
            # 7. הצגת מה שהרחפן רואה על מסך המחשב
            cv2.imshow("Drone Camera Live", processed_img)
            
            # 8. תנאי עצירה: לחיצה על המקש 'q' תעצור הכל
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print(" עצירת חירום הופעלה!")
        
    finally:
        # 9. כיבוי מסודר
        print(" נוחת וסוגר מצלמה...")
        # tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()