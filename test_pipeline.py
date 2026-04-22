import os
import numpy as np
import cv2
import json
from unittest.mock import MagicMock

# Simulate the thermal_monitor logic for verification
def verify_pipeline():
    print("[*] Starting Pipeline Verification...")
    
    # 1. Create a dummy thermal image (just a random numpy array)
    # In reality, this would be a radiometric JPG, but we are testing the processing logic
    dummy_thermal = np.random.uniform(20, 80, (240, 320))
    dummy_thermal[100:110, 150:160] = 95.0  # Force a hotspot
    
    # 2. Mock FlirImageExtractor
    mock_extractor = MagicMock()
    mock_extractor.get_thermal_np.return_value = dummy_thermal
    
    # 3. Simulate processing
    HOTSPOT_THRESHOLD = 70.0
    OUTPUT_DIR = "test_outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"[*] Simulating hotspot detection at {HOTSPOT_THRESHOLD}\u00b0C")
    
    # Find hotspots
    hotspots = np.argwhere(dummy_thermal > HOTSPOT_THRESHOLD)
    print(f"    Found {len(hotspots)} hotspot pixels.")
    
    # Normalize for visualization
    norm_thermal = cv2.normalize(dummy_thermal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_mapped = cv2.applyColorMap(norm_thermal, cv2.COLORMAP_JET)
    
    # Highlight
    for y, x in hotspots:
        cv2.circle(color_mapped, (x, y), 1, (0, 0, 255), -1)
        
    output_path = os.path.join(OUTPUT_DIR, "test_hotspot.png")
    cv2.imwrite(output_path, color_mapped)
    print(f"    [SUCCESS] Visualization saved to {output_path}")
    
    return True

if __name__ == "__main__":
    verify_pipeline()
