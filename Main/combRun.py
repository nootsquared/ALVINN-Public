import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
import time
import torch
import os
import math
# Import saliency model and helper from fastSalGCComb
from Saliency_Predictions.model import fastSal as fastsal
from Saliency_Predictions.utils import load_weight
from Saliency_Predictions.fastSalGCComb import process_frame_saliency
from Metric_Depth_Estimation.metric_depth.alvinn_depth.dpt import ALVINNDepth

# Constants for radar view
RADAR_SIZE = 600  # Size of the radar window
RADAR_CENTER = (RADAR_SIZE // 2, RADAR_SIZE - 50)  # Position of the camera in radar view
MAX_DEPTH = 10.0  # Maximum depth to display in meters
FOV = 60  # Field of view in degrees

def create_radar_frame():
    # Create a black background
    radar = np.zeros((RADAR_SIZE, RADAR_SIZE, 3), dtype=np.uint8)
    
    # Draw radar grid lines
    for r in range(100, RADAR_SIZE, 100):
        cv2.circle(radar, RADAR_CENTER, r, (50, 50, 50), 1)
    
    # Draw FOV lines
    angle = FOV / 2
    end_x1 = int(RADAR_CENTER[0] + RADAR_SIZE * math.sin(math.radians(angle)))
    end_y1 = int(RADAR_CENTER[1] - RADAR_SIZE * math.cos(math.radians(angle)))
    end_x2 = int(RADAR_CENTER[0] - RADAR_SIZE * math.sin(math.radians(angle)))
    end_y2 = int(RADAR_CENTER[1] - RADAR_SIZE * math.cos(math.radians(angle)))
    
    cv2.line(radar, RADAR_CENTER, (end_x1, end_y1), (100, 100, 100), 1)
    cv2.line(radar, RADAR_CENTER, (end_x2, end_y2), (100, 100, 100), 1)
    
    # Draw depth markers
    for d in range(1, int(MAX_DEPTH) + 1):
        y = RADAR_CENTER[1] - int((d / MAX_DEPTH) * (RADAR_SIZE - 100))
        cv2.putText(radar, f"{d}m", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return radar

def update_radar(radar, points):
    # Create a copy of the base radar
    radar_display = radar.copy()
    
    # Draw each point
    for (x, y, depth) in points:
        # Convert depth to radar coordinates
        # Scale depth to radar size (inverse because closer = higher on screen)
        radar_y = RADAR_CENTER[1] - int((depth / MAX_DEPTH) * (RADAR_SIZE - 100))
        
        # Calculate x position based on screen position and FOV
        screen_width = 640  # Assuming standard webcam resolution
        angle = ((x - screen_width/2) / (screen_width/2)) * (FOV/2)
        radar_x = int(RADAR_CENTER[0] + (RADAR_SIZE - 100) * math.sin(math.radians(angle)) * (depth / MAX_DEPTH))
        
        # Draw the point
        cv2.circle(radar_display, (radar_x, radar_y), 5, (0, 0, 255), -1)
        cv2.putText(radar_display, f"{depth:.1f}m", (radar_x + 10, radar_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return radar_display

# ---- Load Models ----
# Saliency Model
saliency_model = fastsal.fastsal(pretrain_mode=False, model_type='A')
sal_weights = load_weight('Saliency_Predictions/weights/SALICON_A.pth', remove_decoder=False)[0]
saliency_model.load_state_dict(sal_weights)
saliency_model.eval()
gpu = torch.cuda.is_available()
if gpu:
    saliency_model.cuda()

# Depth Model
# Set the depth model weight path as needed.
depth_weight_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'Metric_Depth_Estimation', 'checkpoints', 'depth_indoor.pth')
# Use a default config – adjust parameters if necessary.
depth_model = ALVINNDepth(
    encoder='vits',  # Try small ViT
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=80
)
depth_model.load_state_dict(torch.load(depth_weight_path, map_location='cpu'))
depth_model.eval()
DEVICE = 'cuda' if gpu else 'cpu'
depth_model = depth_model.to(DEVICE)
depth_input_size = 518  # as used in the depth code

# Create radar base frame
radar_base = create_radar_frame()

# ---- Start Webcam Loop ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    exit(1)

threshold = 0.65  # initial saliency threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()
    
    # Process saliency for the current frame
    mask, contours, colored_sal = process_frame_saliency(saliency_model, frame, threshold, gpu, probability_output=False)
    
    # Compute depth map for the same frame
    with torch.no_grad():
        depth = depth_model.infer_image(frame, depth_input_size)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    
    # Prepare overlay: blend the colored saliency output with the original frame
    overlay_frame = cv2.addWeighted(frame, 0.75, colored_sal, 0.25, 0)
    
    # Convert mask to grayscale if it's not already
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    # Normalize the mask to 0-255 range
    mask_gray = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply high threshold to get only the brightest splotches
    thresh_value = int(max(0.65, threshold) * 255)  # Minimum threshold of 0.65
    _, thresh = cv2.threshold(mask_gray, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Find contours of the remaining splotches
    splotch_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Store points for radar view
    radar_points = []
    
    # Process each splotch
    for cnt in splotch_contours:
        # Calculate area and skip if too small
        area = cv2.contourArea(cnt)
        if area < 100:  # Increased minimum area to filter out tiny dots
            continue
            
        # Find the center of the splotch
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            
            # Check intensity at the center and surrounding area
            center_intensity = mask_gray[cY, cX]
            if center_intensity < thresh_value * 1.2:  # Must be significantly brighter than threshold
                continue
            
            # Get the depth at this specific point
            point_depth = depth_norm[cY, cX] * MAX_DEPTH  # Scale depth to meters
            
            # Store point for radar view
            radar_points.append((cX, cY, point_depth))
            
            # Draw a circle around the splotch
            radius = int(np.sqrt(area / np.pi))
            radius = max(min(radius, 30), 10)  # Limit radius between 10 and 30 pixels
            
            # Draw the circle
            cv2.circle(overlay_frame, (cX, cY), radius, (0, 255, 0), 2)
            
            # Draw a dot at the center
            cv2.circle(overlay_frame, (cX, cY), 3, (0, 0, 255), -1)
            
            # Display the depth at this specific point
            cv2.putText(overlay_frame, f"{point_depth:.1f}m", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Update and display radar view
    radar_display = update_radar(radar_base, radar_points)
    
    # Display FPS on frame
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(overlay_frame, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    
    # Display both views
    cv2.imshow("Combined Saliency & Depth", overlay_frame)
    cv2.imshow("Radar View", radar_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        threshold = min(1.0, threshold + 0.05)
        print(f"Threshold increased to: {threshold:.2f}")
    elif key == ord('-') or key == ord('_'):
        threshold = max(0.65, threshold - 0.05)
        print(f"Threshold decreased to: {threshold:.2f}")

cap.release()
cv2.destroyAllWindows()