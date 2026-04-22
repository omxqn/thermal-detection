import os
import time
import json
import numpy as np
import cv2
import wmi
from flirimageextractor import FlirImageExtractor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
HOTSPOT_THRESHOLD = 70.0  # Celsius
OUTPUT_DIR = "thermal_outputs"
STATE_FILE = "processed_files.json"
POLL_INTERVAL = 5  # Seconds for USB detection
FLIR_VEN_ID = "26FB" # Standard FLIR Vendor ID

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- STATE MANAGEMENT ---
def load_processed_files():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed):
    with open(STATE_FILE, "w") as f:
        json.dump(list(processed), f)

# --- USB DETECTION ---
def get_flir_drive():
    """Detects the FLIR E8 drive letter on Windows."""
    c = wmi.WMI()
    for drive in c.Win32_DiskDrive():
        # Check InterfaceType and HardwareID or Caption
        if "USB" in drive.InterfaceType:
            # Look for FLIR in Caption or PNPDeviceID
            if "FLIR" in drive.Caption.upper() or FLIR_VEN_ID in drive.PNPDeviceID:
                for partition in drive.associators("Win32_DiskDriveToDiskPartition"):
                    for logical_disk in partition.associators("Win32_LogicalDiskToPartition"):
                        return logical_disk.DeviceID + "\\"
    return None

# --- IMAGE PROCESSING ---
def process_thermal_image(image_path, flir_extractor):
    print(f"[*] Processing: {image_path}")
    try:
        flir_extractor.process_image(image_path)
        # get_thermal_np returns Celsius by default
        thermal_data = flir_extractor.get_thermal_np()
        
        # Stats
        max_t = np.max(thermal_data)
        min_t = np.min(thermal_data)
        avg_t = np.mean(thermal_data)
        
        print(f"    Temp Stats: Max={max_t:.1f}\u00b0C, Min={min_t:.1f}\u00b0C, Avg={avg_t:.1f}\u00b0C")
        
        # Hotspot Detection
        hotspots = np.argwhere(thermal_data > HOTSPOT_THRESHOLD)
        if len(hotspots) > 0:
            print(f"    [!] Detected {len(hotspots)} hotspot pixels!")
        
        # Visualization
        # Normalize thermal data for visualization (0-255)
        norm_thermal = cv2.normalize(thermal_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_mapped = cv2.applyColorMap(norm_thermal, cv2.COLORMAP_JET)
        
        # Highlight hotspots
        for y, x in hotspots:
            cv2.circle(color_mapped, (x, y), 2, (0, 0, 255), -1) # Red dots for hotspots
            
        # Draw stats on image
        label = f"Max: {max_t:.1f}C | Min: {min_t:.1f}C"
        cv2.putText(color_mapped, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save output
        filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")
        cv2.imwrite(output_path, color_mapped)
        print(f"    Saved result to: {output_path}")
        
        return True
    except Exception as e:
        print(f"    [ERROR] Failed to process {image_path}: {e}")
        return False

# --- MONITORING ---
class FlirHandler(FileSystemEventHandler):
    def __init__(self, processed_set, flir_extractor):
        self.processed = processed_set
        self.flir = flir_extractor

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".jpg", ".jpeg")):
            if event.src_path not in self.processed:
                if process_thermal_image(event.src_path, self.flir):
                    self.processed.add(event.src_path)
                    save_processed_files(self.processed)

def start_monitoring(drive_path, processed_set):
    # Standard FLIR storage path
    search_path = os.path.join(drive_path, "DCIM")
    if not os.path.exists(search_path):
        search_path = drive_path # Fallback to root
        
    print(f"[*] Starting live monitor on: {search_path}")
    
    # Process existing files first
    flir = FlirImageExtractor()
    for root, _, files in os.walk(search_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                if full_path not in processed_set:
                    if process_thermal_image(full_path, flir):
                        processed_set.add(full_path)
                        save_processed_files(processed_set)

    # Watchdog setup
    event_handler = FlirHandler(processed_set, flir)
    observer = Observer()
    observer.schedule(event_handler, search_path, recursive=True)
    observer.start()
    return observer

def main():
    processed_files = load_processed_files()
    current_drive = None
    observer = None
    
    print("[*] FLIR E8 Monitor Service Started.")
    print("[*] Waiting for camera connection...")
    
    try:
        while True:
            found_drive = get_flir_drive()
            
            if found_drive and not current_drive:
                print(f"[+] FLIR E8 Connected at {found_drive}")
                current_drive = found_drive
                observer = start_monitoring(current_drive, processed_files)
                
            elif not found_drive and current_drive:
                print("[-] FLIR E8 Disconnected.")
                if observer:
                    observer.stop()
                    observer.join()
                current_drive = None
                observer = None
                
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n[*] Stopping service...")
        if observer:
            observer.stop()
            observer.join()

if __name__ == "__main__":
    main()
