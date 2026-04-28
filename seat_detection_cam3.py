"""
Library Seat Detection - Camera 3
Real-time seat occupancy detection (standalone execution)
- laptop uses low confidence threshold (0.02)
- all others use standard confidence (0.3)
"""

import cv2
import json
import numpy as np
from ultralytics import YOLO

# ==================== Configuration ====================
VIDEO_PATH = 'KakaoTalk_13.mp4'
ROI_FILE = 'seat_roi_cam3.json'
MODEL_PATH = 'yolo11s.pt'

# Per-class confidence thresholds
CLASS_CONFIDENCE = {
    0: 0.3,    # person
    24: 0.3,   # backpack
    56: 0.3,   # chair
    63: 0.02,  # laptop <- low threshold
    67: 0.3    # cell phone
}

HOLD_FRAMES = 30  # Number of frames to hold Occupied status

# Object classes to detect
TARGET_CLASSES = [0, 24, 56, 63, 67]
CLASS_NAMES = {
    0: 'person',
    24: 'backpack',
    56: 'chair',
    63: 'laptop',
    67: 'cell phone'
}

# Color settings
COLORS = {
    'available': (0, 255, 0),
    'occupied': (0, 0, 255),
    'person': (255, 0, 0),
    'object': (0, 255, 255)
}

# ==================== Load ROI ====================
def load_roi():
    """Load ROI data"""
    try:
        with open(ROI_FILE, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            roi = {}
            for seat in data:
                roi[str(seat['id'])] = seat['points']
            return roi
        return data
    except FileNotFoundError:
        print(f"❌ {ROI_FILE} file not found!")
        return {}

# ==================== Utility Functions ====================
def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def box_intersects_polygon(box, polygon):
    """Check if a bounding box overlaps with a polygon"""
    x1, y1, x2, y2 = map(int, box)
    
    points_to_check = [
        ((x1 + x2) // 2, (y1 + y2) // 2),
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
    ]
    
    for point in points_to_check:
        if point_in_polygon(point, polygon):
            return True
    return False

def get_center(box):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def blur_face(frame, box):
    """Apply blur to face region"""
    x1, y1, x2, y2 = map(int, box)
    
    face_height = (y2 - y1) // 3
    face_y2 = y1 + face_height
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    face_y2 = min(frame.shape[0], face_y2)
    
    if x2 > x1 and face_y2 > y1:
        face_region = frame[y1:face_y2, x1:x2]
        blurred = cv2.GaussianBlur(face_region, (51, 51), 30)
        frame[y1:face_y2, x1:x2] = blurred
    
    return frame

# ==================== Main Detection ====================
def main():
    print("\n" + "="*60)
    print("     Camera 3 - Seat Occupancy Detection System")
    print("="*60)
    
    # Load model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Load ROI
    seat_roi = load_roi()
    if not seat_roi:
        return
    print(f"ROI loaded: {len(seat_roi)} seat(s)")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return
    print(f"Video loaded: {VIDEO_PATH}")
    
    print(f"\nPer-class confidence thresholds:")
    for cls_id, conf in CLASS_CONFIDENCE.items():
        print(f"  - {CLASS_NAMES[cls_id]}: {conf}")
    
    print("\n" + "="*60)
    print("Shortcut keys:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("="*60 + "\n")
    
    # State variables
    frame_count = 0
    paused = False
    last_detections = []
    last_seat_status = {seat_id: "Available" for seat_id in seat_roi.keys()}
    hold_counter = {seat_id: 0 for seat_id in seat_roi.keys()}
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            frame_count += 1
            
            if frame_count % 3 == 0:
                # Detect with low conf (0.02) and filter per class
                results = model(frame, conf=0.02, classes=TARGET_CLASSES, verbose=False)
                
                detections = []
                seat_objects = {seat_id: [] for seat_id in seat_roi.keys()}
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for box, cls, conf in zip(boxes, classes, confs):
                        cls = int(cls)
                        
                        # Filter by per-class confidence threshold
                        min_conf = CLASS_CONFIDENCE.get(cls, 0.3)
                        if conf < min_conf:
                            continue  # Skip this detection
                        
                        center = get_center(box)
                        
                        detections.append({
                            'box': box,
                            'class': cls,
                            'conf': conf,
                            'center': center
                        })
                        
                        # Map to seat
                        for seat_id, points in seat_roi.items():
                            if point_in_polygon(center, points) or box_intersects_polygon(box, points):
                                seat_objects[seat_id].append(cls)
                
                # Update seat status
                for seat_id in seat_roi.keys():
                    objects = seat_objects[seat_id]
                    
                    if 0 in objects or any(c in objects for c in [24, 63, 67]):
                        last_seat_status[seat_id] = "Occupied"
                        hold_counter[seat_id] = HOLD_FRAMES
                    else:
                        if hold_counter[seat_id] > 0:
                            hold_counter[seat_id] -= 1
                        else:
                            last_seat_status[seat_id] = "Available"
                
                last_detections = detections
            
            # Apply face blur
            for det in last_detections:
                if det['class'] == 0:
                    frame = blur_face(frame, det['box'])
            
            # Draw bounding boxes
            for det in last_detections:
                box = det['box']
                cls = det['class']
                conf = det['conf']
                x1, y1, x2, y2 = map(int, box)
                
                color = COLORS['person'] if cls == 0 else COLORS['object']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{CLASS_NAMES.get(cls, 'unknown')} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw seat ROI and status
            for seat_id, points in seat_roi.items():
                status = last_seat_status[seat_id]
                color = COLORS['available'] if status == "Available" else COLORS['occupied']
                
                pts = np.array(points, np.int32)
                cv2.polylines(frame, [pts], True, color, 2)
                
                center_x = sum(p[0] for p in points) // 4
                center_y = sum(p[1] for p in points) // 4
                
                text = f"S{seat_id}: {status}"
                cv2.putText(frame, text, (center_x - 40, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display statistics
            available = sum(1 for s in last_seat_status.values() if s == "Available")
            occupied = sum(1 for s in last_seat_status.values() if s == "Occupied")
            
            cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
            cv2.putText(frame, "Library Seat Status", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Total Seats: {len(seat_roi)}", (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Available: {available}", (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Occupied: {occupied}", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Library Seat Detection System - Camera 3', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nProgram terminated.")
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()