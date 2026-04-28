from ultralytics import YOLO
import cv2

model = YOLO('yolo11s.pt')  #  yolo11s

cap = cv2.VideoCapture('KakaoTalk_20260121_12.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    results = model(frame, conf=0.5, classes=[0, 56, 63, 24, 67])
    
    annotated = results[0].plot()
    cv2.imshow('Library Seat Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()