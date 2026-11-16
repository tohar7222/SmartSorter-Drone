import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import torch
import batch_scanner

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_mock = {"known_boxes": []}

def clean_data(data):
    if data is None: return None
    if isinstance(data, (np.integer, int)): return int(data)
    if isinstance(data, (np.floating, float)): return float(data)
    if isinstance(data, np.ndarray): return data.tolist()
    if isinstance(data, torch.Tensor): return data.tolist()
    if isinstance(data, (list, tuple)): return [clean_data(item) for item in data]
    if isinstance(data, dict): return {str(k): clean_data(v) for k, v in data.items()}
    return str(data)

def get_base64_img(cv_image):
    if cv_image is None: return None
    _, buffer = cv2.imencode('.jpg', cv_image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def rgb_to_hex(rgb):
    try:
        if rgb and len(rgb) == 3:
            b, g, r = rgb
            return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))
    except:
        pass
    return '#808080'

@app.post("/init_room")
async def init_room(file: UploadFile = File(...)):
    img_bytes = await file.read()
    temp_input = "temp_input_init.jpg"
    temp_output = "temp_output_init.jpg"
    with open(temp_input, "wb") as f:
        f.write(img_bytes)
    
    _, _, _, _, _, learned_boxes = batch_scanner.process_single_image(temp_input, temp_output, [])
    db_mock["known_boxes"] = learned_boxes
    
    response_boxes = [{"box_id": clean_data(b['id']), "coords": clean_data(b['coords']), "has_toys": bool(b.get('has_toys', False))} for b in learned_boxes] if learned_boxes else []
    return {
        "status": "Init Complete",
        "boxes": response_boxes,
        "processed_image": get_base64_img(cv2.imread(temp_output))
    }

@app.post("/scan_static")
async def scan_static(file: UploadFile = File(...)):
    img_bytes = await file.read()
    temp_input = "temp_input.jpg"
    temp_output = "temp_output.jpg"
    with open(temp_input, "wb") as f:
        f.write(img_bytes)
        
    processed_img, raw_box_id, coords, box_coords, raw_color, current_boxes = batch_scanner.process_single_image(
        temp_input, temp_output, db_mock["known_boxes"]
    )
    
    target_box_id = -1
    try:
        if not isinstance(raw_box_id, (list, tuple, np.ndarray, torch.Tensor)):
            target_box_id = int(float(raw_box_id))
    except:
        pass
    
    # ✨ העדכון החשוב לניהול הזיכרון: מיזוג במקום דריסה
    if current_boxes:
        known_dict = {b['id']: b for b in db_mock["known_boxes"]}
        for new_b in current_boxes:
            known_dict[new_b['id']] = new_b  # מעדכן קואורדינטות או שומר את הקיים
        db_mock["known_boxes"] = list(known_dict.values())

    # מחזירים לאנגולר את כל הקופסאות שיש לנו בזיכרון, ולא רק את אלו שזוהו בשנייה האחרונה
    response_boxes = [{"box_id": clean_data(b['id']), "coords": clean_data(b['coords'])} for b in db_mock["known_boxes"]] if db_mock["known_boxes"] else []
    
    return {
        "status": "Static Scan Complete",
        "box_id": clean_data(target_box_id),
        "target_toy_coords": clean_data(coords),
        "target_box_coords": clean_data(box_coords),
        "box_color": rgb_to_hex(raw_color),
        "current_boxes": response_boxes,
        "processed_image": get_base64_img(cv2.imread(temp_output))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)