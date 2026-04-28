"""
SeatSense - Library Seat Occupancy Detection System
Real-time seat monitoring across 3 campuses using YOLO11
Includes live video streaming with annotated frames
"""

from flask import Flask, render_template, jsonify, Response
from ultralytics import YOLO
import cv2
import json
import threading
import time
import numpy as np
from database import (
    init_database, log_occupancy,
    get_hourly_stats, get_peak_hours, get_seat_popularity,
    get_daily_summary, get_current_vs_average, get_total_records
)

app = Flask(__name__)

# ==================== Global Variables ====================
model = YOLO('yolo11s.pt')

cameras = {
    'cam1': {
        'video_path': 'KakaoTalk_11.mp4',
        'roi_file': 'seat_roi.json',
        'name': 'Camera 1',
        'seat_roi': {},
        'seat_status': {},
        'cap': None,
        'confidence': 0.3,
        'laptop_conf': 0.1,
        'use_box_intersect': False,
        'hold_frames': 0
    },
    'cam2': {
        'video_path': 'KakaoTalk_12.mp4',
        'roi_file': 'seat_roi_cam2.json',
        'name': 'Camera 2',
        'seat_roi': {},
        'seat_status': {},
        'cap': None,
        'confidence': 0.3,
        'laptop_conf': 0.3,
        'use_box_intersect': False,
        'hold_frames': 0
    },
    'cam3': {
        'video_path': 'KakaoTalk_13.mp4',
        'roi_file': 'seat_roi_cam3.json',
        'name': 'Camera 3',
        'seat_roi': {},
        'seat_status': {},
        'cap': None,
        'confidence': 0.3,    # person/backpack/cellphone threshold
        'laptop_conf': 0.02,  # laptop needs very low conf to detect
        'use_box_intersect': True,
        'hold_frames': 30
    }
}

campuses = {
    'main_wing': {'name': 'Main Wing', 'cameras': ['cam1', 'cam3']},
    'stem_wing': {'name': 'STEM Wing', 'cameras': ['cam1', 'cam2', 'cam3']},
    'cathay':    {'name': 'Cathay',    'cameras': ['cam2', 'cam3']}
}

TARGET_CLASSES = [0, 56, 63, 24, 67]
CLASS_NAMES = {0: 'person', 24: 'backpack', 56: 'chair', 63: 'laptop', 67: 'cell phone'}

# Per-seat hold frames - only seat 8 has a permanent laptop that YOLO occasionally misses
SEAT_HOLD_FRAMES = {
    8: 90   # ~9 seconds hold for seat 8
}

detection_running = False
DB_LOG_INTERVAL = 60

# ==================== Frame Buffers ====================
# raw_frames         : latest decoded frame (updated at native FPS by reader thread)
# latest_detections  : last known YOLO detections per camera (updated by detector)
raw_frames        = {cam_id: None for cam_id in cameras}
latest_detections = {cam_id: [] for cam_id in cameras}
raw_locks         = {cam_id: threading.Lock() for cam_id in cameras}
det_locks         = {cam_id: threading.Lock() for cam_id in cameras}

# Keep latest_frames for backward compat but generate_frames won't use it
latest_frames = {cam_id: None for cam_id in cameras}
frame_locks   = {cam_id: threading.Lock() for cam_id in cameras}


# ==================== ROI Functions ====================
def load_roi_data():
    """Load ROI data for all cameras from JSON files."""
    for cam_id, cam_config in cameras.items():
        try:
            with open(cam_config['roi_file'], 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                cam_config['seat_roi'] = {str(s['id']): s['points'] for s in data}
            else:
                cam_config['seat_roi'] = data
            print(f"[OK] {cam_id} ROI loaded: {len(cam_config['seat_roi'])} seats")
        except FileNotFoundError:
            print(f"[WARN] {cam_config['roi_file']} not found")
            cam_config['seat_roi'] = {}


def point_in_polygon(point, polygon):
    """Ray casting algorithm."""
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
    """Check if bounding box corners/centre intersect a polygon (Camera 3)."""
    x1, y1, x2, y2 = map(int, box)
    check = [((x1+x2)//2, (y1+y2)//2), (x1,y1), (x2,y1), (x1,y2), (x2,y2)]
    return any(point_in_polygon(p, polygon) for p in check)


def get_box_center(box):
    x1, y1, x2, y2 = box
    return (int((x1+x2)/2), int((y1+y2)/2))


# ==================== Frame Annotation ====================
def blur_faces(frame, detections):
    """Blur upper third of each detected person for privacy."""
    for det in detections:
        if det['class'] == 0:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            fy2 = y1 + (y2 - y1) // 3
            fy1, fy2 = max(0, y1), min(frame.shape[0], fy2)
            fx1, fx2 = max(0, x1), min(frame.shape[1], x2)
            if fy2 > fy1 and fx2 > fx1:
                frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(frame[fy1:fy2, fx1:fx2], (51,51), 30)
    return frame


def draw_annotations(frame, cam_id, detections):
    """Draw ROI polygons, seat status labels, bounding boxes and summary."""
    cam_config = cameras[cam_id]
    seat_roi    = cam_config['seat_roi']
    seat_status = cam_config['seat_status']

    # Draw seat ROI polygons + labels
    for seat_id, points in seat_roi.items():
        status = seat_status.get(seat_id, 'Available')
        color  = (0, 0, 255) if status == 'Occupied' else (0, 255, 0)

        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, 2)

        cx = sum(p[0] for p in points) // len(points)
        cy = sum(p[1] for p in points) // len(points)
        label = f'S{seat_id}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (cx-tw//2-3, cy-th-3), (cx+tw//2+3, cy+3), color, -1)
        cv2.putText(frame, label, (cx-tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        cls   = det['class']
        color = (255, 0, 0) if cls == 0 else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det['class_name']} {det['conf']:.2f}",
                    (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Summary overlay (top-left)
    total    = len(seat_status)
    occupied = sum(1 for s in seat_status.values() if s == 'Occupied')
    available = total - occupied

    cv2.rectangle(frame, (8, 8), (220, 100), (0,0,0), -1)
    cv2.rectangle(frame, (8, 8), (220, 100), (200,200,200), 1)
    cv2.putText(frame, f'SeatSense | {cam_config["name"]}',
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f'Total:     {total}',
                (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
    cv2.putText(frame, f'Available: {available}',
                (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,0), 1)
    cv2.putText(frame, f'Occupied:  {occupied}',
                (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,100,255), 1)

    return frame


# ==================== Database Logging ====================
def db_logging_loop():
    global detection_running
    while detection_running:
        time.sleep(DB_LOG_INTERVAL)
        for cam_id, cam_config in cameras.items():
            if cam_config['seat_status']:
                log_occupancy(cam_id, cam_config['seat_status'])
        print(f"[DB] Logged at {time.strftime('%H:%M:%S')}")


# ==================== Detection Loop ====================
def frame_reader_loop(cam_id):
    """
    Thread 1 (per camera): reads frames at native FPS and stores in raw_frames.
    No YOLO here - keeps live stream smooth regardless of detection speed.
    """
    global detection_running

    cam_config = cameras[cam_id]
    cap = cv2.VideoCapture(cam_config['video_path'])
    cam_config['cap'] = cap

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video for {cam_id}")
        return

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / src_fps
    print(f"[OK] {cam_id} reader started (src_fps={src_fps:.1f})")

    while detection_running:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        with raw_locks[cam_id]:
            raw_frames[cam_id] = frame

        elapsed = time.time() - t_start
        sleep_t = frame_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    cap.release()
    print(f"[OK] {cam_id} reader stopped")

hold_counters = {}

def detection_loop(cam_id):
    global detection_running, hold_counters

    cam_config = cameras[cam_id]
    cap = cv2.VideoCapture(cam_config['video_path'])
    cam_config['cap'] = cap

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video for {cam_id}")
        return

    base_conf   = cam_config['confidence']
    laptop_conf = cam_config['laptop_conf']
    use_box_int = cam_config['use_box_intersect']

    # Get source FPS to maintain real-time playback speed
    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / src_fps  # target seconds per frame

    print(f"[OK] {cam_id} started (conf={base_conf}, src_fps={src_fps:.1f})")
    hold_counters[cam_id] = {sid: 0 for sid in cam_config['seat_roi']}  # per-seat hold counters
    frame_count     = 0
    last_detections = []
    last_time       = time.time()

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_time = time.time()
            continue

        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        frame_count += 1

        if frame_count % 3 == 0:
            results = model(frame, conf=min(base_conf, laptop_conf),
                            classes=TARGET_CLASSES, verbose=False)

            last_detections = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls  = int(box.cls[0].cpu().numpy())
                    min_c = laptop_conf if cls == 63 else base_conf
                    if conf >= min_c:
                        last_detections.append({
                            'box': [x1,y1,x2,y2],
                            'conf': conf,
                            'class': cls,
                            'class_name': CLASS_NAMES.get(cls, 'unknown')
                        })

            # Share detections with generate_frames for real-time stream
            with det_locks[cam_id]:
                latest_detections[cam_id] = list(last_detections)

            # Map detections to seats
            seat_objects = {sid: {'person': False, 'belongings': False}
                            for sid in cam_config['seat_roi']}

            for det in last_detections:
                center = get_box_center(det['box'])
                for seat_id, points in cam_config['seat_roi'].items():
                    in_seat = (point_in_polygon(center, points) or
                               box_intersects_polygon(det['box'], points)) if use_box_int \
                              else point_in_polygon(center, points)
                    if in_seat:
                        if det['class'] == 0:
                            seat_objects[seat_id]['person'] = True
                        elif det['class'] in [63, 24, 67]:
                            seat_objects[seat_id]['belongings'] = True
                        # No break - one detection can match multiple seats (same as seat_detection_cam3.py)

            # Update seat status with hold logic
            # Per-seat SEAT_HOLD_FRAMES takes priority; fallback to camera hold_frames setting
            cam_hold_default = cam_config['hold_frames']
            for seat_id in cam_config['seat_roi']:
                is_occupied = seat_objects[seat_id]['person'] or seat_objects[seat_id]['belongings']
                seat_hold = SEAT_HOLD_FRAMES.get(int(seat_id), cam_hold_default)

                if is_occupied:
                    cam_config['seat_status'][seat_id] = 'Occupied'
                    if seat_hold > 0:
                        hold_counters[cam_id][seat_id] = seat_hold
                else:
                    if seat_hold > 0 and hold_counters[cam_id][seat_id] > 0:
                        hold_counters[cam_id][seat_id] -= 1
                        cam_config['seat_status'][seat_id] = 'Occupied'
                    else:
                        cam_config['seat_status'][seat_id] = 'Available'

        # Annotate and push to latest_frames for streaming (same as seat_detection.py)
        display = blur_faces(frame, last_detections)
        display = draw_annotations(display, cam_id, last_detections)

        with frame_locks[cam_id]:
            latest_frames[cam_id] = display

    cap.release()
    print(f"[OK] {cam_id} detector stopped")


# ==================== Live Stream Generator ====================
def generate_frames(cam_id):
    """MJPEG frame generator - streams YOLO-annotated frames directly."""
    while True:
        with frame_locks[cam_id]:
            frame = latest_frames[cam_id]

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)


# ==================== Flask Routes ====================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    if cam_id not in cameras:
        return "Camera not found", 404
    return Response(generate_frames(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ==================== Campus API ====================
@app.route('/api/campuses')
def get_campuses():
    result = []
    for campus_id, campus_config in campuses.items():
        total_seats = sum(len(cameras[c]['seat_roi']) for c in campus_config['cameras'])
        result.append({
            'id': campus_id,
            'name': campus_config['name'],
            'camera_count': len(campus_config['cameras']),
            'total_seats': total_seats
        })
    return jsonify(result)


@app.route('/api/campus/<campus_id>')
def get_campus_data(campus_id):
    if campus_id not in campuses:
        return jsonify({'error': 'Campus not found'}), 404

    campus = campuses[campus_id]
    camera_data = []
    total_seats = total_available = total_occupied = 0

    for cam_id in campus['cameras']:
        cam   = cameras[cam_id]
        seats = {f"Seat {k}": v for k, v in cam['seat_status'].items()}
        avail = sum(1 for s in seats.values() if s == 'Available')
        occ   = sum(1 for s in seats.values() if s == 'Occupied')

        camera_data.append({'id': cam_id, 'name': cam['name'],
                             'seats': seats, 'available': avail, 'occupied': occ})
        total_seats     += len(seats)
        total_available += avail
        total_occupied  += occ

    return jsonify({'id': campus_id, 'name': campus['name'],
                    'cameras': camera_data,
                    'total_seats': total_seats,
                    'total_available': total_available,
                    'total_occupied': total_occupied})


@app.route('/api/seats')
def get_seats():
    result = {}
    for cam_id, cam_config in cameras.items():
        formatted = {f"Seat {k}": v for k, v in cam_config['seat_status'].items()}
        result[cam_id] = {'name': cam_config['name'], 'seats': formatted}
    return jsonify(result)


@app.route('/api/stats')
def get_stats():
    total = available = occupied = 0
    for cam_config in cameras.values():
        for status in cam_config['seat_status'].values():
            total += 1
            if status == 'Available':
                available += 1
            else:
                occupied += 1
    return jsonify({'total': total, 'available': available, 'occupied': occupied})


@app.route('/api/cameras')
def get_cameras_api():
    return jsonify([
        {'id': cid, 'name': c['name'], 'seat_count': len(c['seat_roi'])}
        for cid, c in cameras.items()
    ])


# ==================== Analytics API ====================
@app.route('/api/analytics/hourly')
def api_hourly_stats():
    return jsonify(get_hourly_stats())

@app.route('/api/analytics/hourly/<date>')
def api_hourly_stats_date(date):
    return jsonify(get_hourly_stats(date))

@app.route('/api/analytics/peak')
def api_peak_hours():
    return jsonify(get_peak_hours())

@app.route('/api/analytics/seat-popularity')
def api_seat_popularity():
    return jsonify(get_seat_popularity())

@app.route('/api/analytics/daily')
def api_daily_summary():
    return jsonify(get_daily_summary())

@app.route('/api/analytics/current-vs-average')
def api_current_vs_average():
    return jsonify(get_current_vs_average())

@app.route('/api/analytics/total-records')
def api_total_records():
    return jsonify({'total_records': get_total_records()})


# ==================== Startup ====================
if __name__ == '__main__':
    init_database()
    load_roi_data()

    has_cameras = any(cam['seat_roi'] for cam in cameras.values())

    if not has_cameras:
        print("[WARN] No ROI data found.")
    else:
        for cam_id, cam_config in cameras.items():
            for seat_id in cam_config['seat_roi']:
                cam_config['seat_status'][seat_id] = 'Available'

        detection_running = True

        for cam_id in cameras:
            if cameras[cam_id]['seat_roi']:
                # Start reader thread first (native FPS), then detector thread (YOLO)
                threading.Thread(target=frame_reader_loop, args=(cam_id,), daemon=True).start()
                threading.Thread(target=detection_loop,    args=(cam_id,), daemon=True).start()

        threading.Thread(target=db_logging_loop, daemon=True).start()

        print("\n" + "=" * 50)
        print("  SeatSense - Library Seat Detection System")
        print("=" * 50)
        for cam_id, cam in cameras.items():
            print(f"  {cam['name']}: {len(cam['seat_roi'])} seats "
                  f"(conf={cam['confidence']}, hold={cam['hold_frames']})")
        print(f"\n  Dashboard : http://localhost:5000")
        print(f"  Live Feed : http://localhost:5000/live")
        print("=" * 50 + "\n")

        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)