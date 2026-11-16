from ultralytics import YOLO

if __name__ == '__main__':
    # מתחילים ממודל בסיס חכם ומהיר של YOLO
    model = YOLO('yolov8n.pt') 

    # הנתיב לקובץ הנתונים שמעודכן עם 3 המחלקות
    yaml_path = r"C:\טוהר בלולו\פרויקטם\drone_project\data.yaml"
    
    # מתחילים אימון טרי וחזק על כל הנתונים שלנו!
    results = model.train(
        data=yaml_path, 
        epochs=50,      
        imgsz=640,      
        device='cpu',   
        workers=0,      
        
        plots=True
    )
    
    print("האימון הסתיים בהצלחה! הרחפן שלך עכשיו חכם הרבה יותר. 🚀")