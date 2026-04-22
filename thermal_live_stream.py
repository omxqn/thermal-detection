import cv2
import numpy as np
import time
import os

# --- CONFIGURATION ---
CAMERA_INDEX = 0      # Try 0, 1, or 2
BRIGHTNESS_THRESHOLD = 230 # 0-255, threshold for "hot" areas in visual feed
MIN_HOTSPOT_AREA = 50      # Minimum number of pixels to be considered a hotspot
OUTPUT_DIR = "live_hotspots"
SNAPSHOT_INTERVAL = 2      # Seconds between saving snapshots if hotspot persists

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def start_live_monitor():
    print(f"[*] Starting FLIR Live Stream Monitor on Index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        print("Tip: Check if another app (like Windows Camera) is using it.")
        return

    last_snapshot_time = 0
    
    print("[*] Monitoring for hotspots (Visual Brightness Method)...")
    print("[*] Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[-] Lost connection to camera.")
                break

            # 1. Processing for Hotspot Detection
            # Convert to grayscale to find brightest areas
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to find hotspots
            _, thresh = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # Clean up noise
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours of hotspots
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            hotspot_found = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_HOTSPOT_AREA:
                    hotspot_found = True
                    # Draw bounding box
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "HOT!", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 2. Logic for saving snapshots
            current_time = time.time()
            if hotspot_found and (current_time - last_snapshot_time > SNAPSHOT_INTERVAL):
                filename = f"hotspot_{time.strftime('%Y%p%m_%H%M%S')}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, frame)
                print(f"    [!] Hotspot captured: {filename}")
                last_snapshot_time = current_time

            # 3. UI Overlays
            status_text = "Status: Monitoring..."
            if hotspot_found:
                status_text = "Status: HOTSPOT DETECTED!"
            
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not hotspot_found else (0, 0, 255), 2)
            cv2.putText(frame, "FLIR E8 Live Stream", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 4. Display
            cv2.imshow('FLIR Monitoring Pipeline', frame)

            # Exit logic
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Monitor stopped.")

if __name__ == "__main__":
    start_live_monitor()
