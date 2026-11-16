import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO

print("טוען מודלים של YOLO...")
box_model = YOLO('boxes_model.pt')
toy_model = YOLO('toys_model.pt')

tello = Tello()
tello.connect()
print(f"סוללה: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()
print("מצלמה הופעלה. מחכה 5 שניות להתארגנות...")
time.sleep(5)

print("ממריא!")
tello.takeoff()
tello.move_up(50) 

print("מתחיל סריקה... לחצי 'q' כדי לסיים ולנחות.")
while True:
    frame = frame_read.frame
    
    box_results = box_model(frame, stream=True)
    toy_results = toy_model(frame, stream=True)
    
    for r in box_results:
        frame = r.plot()
        
    for r in toy_results:
        frame = r.plot()
        
    cv2.imshow("Smart Sorter - Drone View", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("נוחת...")
tello.land()
tello.streamoff()
cv2.destroyAllWindows()