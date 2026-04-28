import cv2
import json

# Global variables
seats = []
current_seat = []
frame = None
original_frame = None

def mouse_callback(event, x, y, flags, param):
    """Handle mouse click events"""
    global current_seat, seats, frame
    
    # Left click: add coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        current_seat.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Display coordinate number
        cv2.putText(frame, str(len(current_seat)), (x+10, y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('ROI Setup - Camera 2', frame)
        
        # Complete rectangle when 4 coordinates are collected
        if len(current_seat) == 4:
            seats.append({
                'id': len(seats) + 1,
                'points': current_seat.copy()
            })
            
            # Draw rectangle
            for i in range(4):
                cv2.line(frame, current_seat[i], current_seat[(i+1)%4], (0, 255, 0), 2)
            
            # Display seat number
            center_x = sum(p[0] for p in current_seat) // 4
            center_y = sum(p[1] for p in current_seat) // 4
            cv2.putText(frame, f'Seat {len(seats)}', (center_x-30, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('ROI Setup - Camera 2', frame)
            current_seat = []
            print(f"✓ Seat {len(seats)} saved!")

def setup_roi(video_path):
    """Main function for ROI setup"""
    global frame, original_frame, seats
    
    # Read first frame of video
    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    
    if not ret:
        print("Cannot read video!")
        print(f"Please check the file path: {video_path}")
        return
    
    # Resize resolution
    original_frame = cv2.resize(original_frame, (0, 0), fx=0.5, fy=0.5)
    frame = original_frame.copy()
    
    # Window setup
    cv2.namedWindow('ROI Setup - Camera 2')
    cv2.setMouseCallback('ROI Setup - Camera 2', mouse_callback)
    
    # Print instructions
    print("\n" + "="*60)
    print("        Seat ROI Setup Program (Camera 2)")
    print("="*60)
    print("\nHow to use:")
    print("  1. Click 4 corners for each seat")
    print("  2. Click order: Top-left -> Top-right -> Bottom-right -> Bottom-left")
    print("     (Clockwise!)")
    print("\nShortcut keys:")
    print("  's' - Save and exit")
    print("  'r' - Reset (start over)")
    print("  'u' - Delete last seat (Undo)")
    print("  'q' - Exit without saving")
    print("\nTips:")
    print("  - Define seat boundaries clearly")
    print("  - Make sure seats do not overlap")
    print("="*60 + "\n")
    
    while True:
        cv2.imshow('ROI Setup - Camera 2', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Save and exit
        if key == ord('s'):
            if len(seats) > 0:
                with open('seat_roi_cam2.json', 'w') as f:
                    json.dump(seats, f, indent=2)
                print(f"\n{len(seats)} seat(s) saved to 'seat_roi_cam2.json'!")
                print(f"Saved seats: ", end="")
                for i, seat in enumerate(seats, 1):
                    print(f"Seat {i}", end=", " if i < len(seats) else "\n")
            else:
                print("\nNo seats to save!")
            break
        
        # Reset
        elif key == ord('r'):
            seats = []
            current_seat = []
            frame = original_frame.copy()
            cv2.imshow('ROI Setup - Camera 2', frame)
            print("\nReset complete!")
        
        # Delete last seat (Undo)
        elif key == ord('u'):
            if len(seats) > 0:
                removed_seat = seats.pop()
                frame = original_frame.copy()
                # Redraw remaining seats
                for seat in seats:
                    points = seat['points']
                    for i in range(4):
                        cv2.line(frame, points[i], points[(i+1)%4], (0, 255, 0), 2)
                    center_x = sum(p[0] for p in points) // 4
                    center_y = sum(p[1] for p in points) // 4
                    cv2.putText(frame, f'Seat {seat["id"]}', (center_x-30, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('ROI Setup - Camera 2', frame)
                print(f"↩️  Seat {removed_seat['id']} removed (remaining seats: {len(seats)})")
            else:
                print("No seats to delete!")
        
        # Exit
        elif key == ord('q'):
            print("\nExiting without saving.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Camera 2 video file path
    video_path = 'KakaoTalk_12.mp4'
    
    print("Video file:", video_path)
    print("Camera 2 ROI Setup")
    setup_roi(video_path)