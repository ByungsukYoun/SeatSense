import cv2
import json
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo11s.pt')

CLASS_CONFIDENCE = {
    0: 0.3,    # person
    24: 0.3,   # backpack
    56: 0.5,   # chair
    63: 0.16,  # laptop
    67: 0.3    # cell phone
}

# Per-seat hold frames: keep Occupied status for N processed frames after last detection
# Only applied to seats where detection is unreliable (e.g. seat 8 has a permanent laptop)
SEAT_HOLD_FRAMES = {
    8: 90   # ~9 seconds - laptop on seat 8 is occasionally missed by YOLO
    # Add other seat IDs here if needed, e.g.  3: 30
}
DEFAULT_HOLD_FRAMES = 0  # All other seats: no hold (instant Available)

CLASS_NAMES = {0: 'person', 24: 'backpack', 56: 'chair', 63: 'laptop', 67: 'cell phone'}


def load_roi():
    """Load ROI data from JSON file."""
    try:
        with open('seat_roi.json', 'r') as f:
            seats = json.load(f)
        print(f"Loaded {len(seats)} seat ROIs.")
        return seats
    except FileNotFoundError:
        print("seat_roi.json not found. Run roi_setup.py first.")
        return None


def point_in_polygon(point, polygon):
    """Ray casting algorithm to check if a point is inside a polygon."""
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


def get_box_center(box):
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def blur_faces(frame, detections):
    """Blur the upper third of each detected person (privacy protection)."""
    for det in detections:
        if det['class'] == 0:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            face_y2 = y1 + (y2 - y1) // 3
            fy1 = max(0, y1)
            fy2 = min(frame.shape[0], face_y2)
            fx1 = max(0, x1)
            fx2 = min(frame.shape[1], x2)
            if fy2 > fy1 and fx2 > fx1:
                face_region = frame[fy1:fy2, fx1:fx2]
                frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(face_region, (51, 51), 30)
    return frame


def determine_seat_status(seats, detections):
    """Determine raw seat status from current detections (without hold logic)."""
    seat_status = {}
    for seat in seats:
        seat_status[seat['id']] = {
            'status': 'Available',
            'has_person': False,
            'has_belongings': False,
            'objects': []
        }

    for det in detections:
        box = det['box']
        cls = det['class']
        class_name = det['class_name']
        center = get_box_center(box)

        for seat in seats:
            if point_in_polygon(center, seat['points']):
                seat_id = seat['id']
                seat_status[seat_id]['objects'].append(class_name)
                if cls == 0:
                    seat_status[seat_id]['has_person'] = True
                elif cls in [63, 24, 67]:
                    seat_status[seat_id]['has_belongings'] = True
                break

    for seat_id in seat_status:
        if seat_status[seat_id]['has_person'] or seat_status[seat_id]['has_belongings']:
            seat_status[seat_id]['status'] = 'Occupied'

    return seat_status


def apply_hold_logic(seat_status, hold_counters):
    """
    Apply per-seat hold frame logic to prevent flickering.
    Only seats listed in SEAT_HOLD_FRAMES are affected.
    All other seats switch to Available immediately.
    """
    for seat_id in seat_status:
        hold = SEAT_HOLD_FRAMES.get(seat_id, DEFAULT_HOLD_FRAMES)
        if hold == 0:
            continue  # No hold for this seat - leave status as-is

        if seat_status[seat_id]['status'] == 'Occupied':
            hold_counters[seat_id] = hold  # Reset counter on detection
        else:
            if hold_counters[seat_id] > 0:
                hold_counters[seat_id] -= 1
                seat_status[seat_id]['status'] = 'Occupied'  # Hold as Occupied

    return seat_status, hold_counters


def draw_seats_and_status(frame, seats, seat_status):
    """Draw seat ROI polygons and status labels on the frame."""
    for seat in seats:
        seat_id = seat['id']
        points = seat['points']
        status = seat_status[seat_id]['status']

        color = (0, 0, 255) if status == 'Occupied' else (0, 255, 0)
        status_text = status

        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, 2)

        cx = sum(p[0] for p in points) // 4
        cy = sum(p[1] for p in points) // 4

        text = f'S{seat_id}: {status_text}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (cx - tw//2 - 5, cy - th - 5), (cx + tw//2 + 5, cy + 5), color, -1)
        cv2.putText(frame, text, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_summary(frame, seat_status):
    """Draw occupancy summary in the top-left corner."""
    total = len(seat_status)
    occupied = sum(1 for s in seat_status.values() if s['status'] == 'Occupied')
    available = total - occupied

    cv2.rectangle(frame, (10, 10), (250, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (250, 110), (255, 255, 255), 2)
    cv2.putText(frame, 'SeatSense - Camera 1', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Total Seats: {total}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'Available:   {available}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'Occupied:    {occupied}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def main():
    seats = load_roi()
    if seats is None:
        return

    cap = cv2.VideoCapture('KakaoTalk_11.mp4')
    if not cap.isOpened():
        print("Cannot open video file.")
        return

    print("\n" + "=" * 60)
    print("  SeatSense - Seat Occupancy Detection (Camera 1)")
    print("=" * 60)
    print(f"\nPer-seat hold frames: {SEAT_HOLD_FRAMES} (others: instant)")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause / Resume")
    print("=" * 60 + "\n")

    paused = False
    frame_count = 0
    last_detections = []
    last_seat_status = None
    current_frame = None

    # Initialise hold counters (only seats with SEAT_HOLD_FRAMES > 0 will use them)
    hold_counters = {seat['id']: 0 for seat in seats}

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frame_count += 1

            if frame_count % 3 == 0 or last_seat_status is None:
                results = model(frame, conf=0.07, classes=[0, 56, 63, 24, 67], verbose=False)

                detections = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        if conf < CLASS_CONFIDENCE.get(cls, 0.3):
                            continue
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'conf': conf,
                            'class': cls,
                            'class_name': CLASS_NAMES.get(cls, 'unknown')
                        })

                last_detections = detections
                raw_status = determine_seat_status(seats, detections)

                # Apply hold logic to fix flickering
                last_seat_status, hold_counters = apply_hold_logic(raw_status, hold_counters)

            frame = blur_faces(frame, last_detections)

            if last_seat_status:
                draw_seats_and_status(frame, seats, last_seat_status)
                draw_summary(frame, last_seat_status)

            # Draw bounding boxes
            for det in last_detections:
                x1, y1, x2, y2 = [int(v) for v in det['box']]
                cls = det['class']
                color = (255, 0, 0) if cls == 0 else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{det['class_name']} {det['conf']:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            current_frame = frame.copy()
        else:
            if current_frame is not None:
                frame = current_frame.copy()

        cv2.imshow('SeatSense - Camera 1', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == '__main__':
    main()