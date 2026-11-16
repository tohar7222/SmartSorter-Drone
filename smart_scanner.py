import cv2
import numpy as np

# הנתיב לתמונת המבחן (תדביקי פה את הנתיב האמיתי של התמונה שלך)
image_path = r"C:\טוהר בלולו\פרויקטם\סירטוני גן\training_frame_476.jpg"
# קורא את התמונה בעזרת numpy כדי לעקוף את בעיית העברית
img_array = np.fromfile(image_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
img_copy = img.copy()

# המרת התמונה לגווני אפור וטשטוש כדי להתעלם מרעשים קטנים
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# מציאת קצוות (מפריד את כל החפצים מהרצפה)
edges = cv2.Canny(blurred, 30, 100)

# "עיבוי" קל כדי לחבר חלקים של אותו חפץ
kernel = np.ones((5,5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)

# מציאת כל הצורות/הכתמים בתמונה
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# הגדרת חוקי הגודל שלך! (אפשר לשחק עם המספרים האלה אחר כך)
MIN_TOY_SIZE = 400    # קטן מזה? זה סתם לכלוך או קו של בלטה
MAX_TOY_SIZE = 15000  # גדול מזה? זה רהיט (ספה, שולחן, שטיח)

toys_found = 0

print("מתחיל לסרוק ולסנן לפי חוקי הגודל שלך...")

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    # הנה הלוגיקה שלך בפעולה:
    if area > MAX_TOY_SIZE:
        # זה רהיט - נסמן באדום רק כדי שנראה שהקוד זיהה והתעלם
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_copy, "FURNITURE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    elif MIN_TOY_SIZE < area < MAX_TOY_SIZE:
        # בינגו! עבר את שתי המסננות - זה צעצוע!
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(img_copy, "TOY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        toys_found += 1

print(f"הסריקה הסתיימה! מצאתי {toys_found} צעצועים שמתאימים בגודל.")

output_path = r"C:\טוהר בלולו\פרויקטם\drone_project\SCAN_TEST\size_filter_result.jpg"
cv2.imwrite(output_path, img_copy)