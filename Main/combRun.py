import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
import time
import torch
import os
# Import saliency model and helper from fastSalGCComb
from Saliency_Predictions.model import fastSal as fastsal
from Saliency_Predictions.utils import load_weight
from Saliency_Predictions.fastSalGCComb import process_frame_saliency
from Metric_Depth_Estimation.metric_depth.alvinn_depth.dpt import ALVINNDepth

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

# Variables for temporal stabilization
prev_depth = None
prev_frame = None
stabilization_enabled = True
consistency_threshold = 0.1
max_stabilization = 0.7
motion_threshold = 30.0

# ---- Start Webcam Loop ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    exit(1)

threshold = 0.5  # initial saliency threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()
    
    # Process saliency for the current frame
    mask, contours, colored_sal = process_frame_saliency(saliency_model, frame, threshold, gpu, probability_output=False)
    
    # Additional processing to separate close hotspots
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.erode(mask, kernel, iterations=1)  # Erode to separate close spots
    
    # Find contours with RETR_EXTERNAL for only outer contours
    refined_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compute depth map for the same frame
    with torch.no_grad():
        depth = depth_model.infer_image(frame, depth_input_size)
    
    # Apply temporal stabilization if enabled
    if stabilization_enabled and prev_depth is not None:
        # Calculate motion between frames
        if prev_frame is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            motion = np.mean(np.abs(frame_gray.astype(float) - prev_gray.astype(float)))
            
            # Apply stabilization when motion is below threshold
            if motion < motion_threshold:
                # Calculate stability weight based on motion
                stability_weight = max_stabilization * (1 - motion / motion_threshold)
                
                # Blend current and previous depth
                depth = (1 - stability_weight) * depth + stability_weight * prev_depth
    
    # Store current frame and depth for next iteration
    prev_frame = frame.copy()
    prev_depth = depth.copy()
    
    # Convert to metric depth values (meters)
    depth_meters = depth * depth_model.max_depth / 100
    
    # Normalize depth for visualization
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    
    # Prepare overlay: blend the colored saliency output with the original frame
    overlay_frame = cv2.addWeighted(frame, 0.75, colored_sal, 0.25, 0)
    
    # For each contour, compute average depth within that blob
    for cnt in refined_contours:  # Use the refined contours
        if cv2.contourArea(cnt) < 50:
            continue  # ignore tiny blobs
        
        # Create a blank mask for the contour
        contour_mask = np.zeros(depth_norm.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 1, thickness=-1)
        
        # Compute average depth in meters
        if np.sum(contour_mask) > 0:
            avg_depth_norm = np.sum(depth_norm * contour_mask) / np.sum(contour_mask)
            avg_depth_meters = np.sum(depth_meters * contour_mask) / np.sum(contour_mask)
        else:
            avg_depth_norm = 0
            avg_depth_meters = 0
            
        # Get contour centroid
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = cnt[0][0]
        
        # Overlay text showing metric depth in meters
        cv2.putText(overlay_frame, f"{avg_depth_meters:.2f}m", (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        
        # Create a filled green contour with partial transparency
        contour_fill = np.zeros_like(overlay_frame)
        cv2.drawContours(contour_fill, [cnt], -1, (0,255,0), -1)
        overlay_frame = cv2.addWeighted(overlay_frame, 1.0, contour_fill, 0.3, 0)
        
        # Outline the contour in green
        cv2.drawContours(overlay_frame, [cnt], -1, (0,255,0), 2)
    
    # Display stabilization status
    stab_status = "ON" if stabilization_enabled else "OFF"
    cv2.putText(overlay_frame, f"Stab: {stab_status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    
    # Display FPS on frame
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(overlay_frame, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow("Combined Saliency & Depth", overlay_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        threshold = min(1.0, threshold + 0.05)
        print(f"Threshold increased to: {threshold:.2f}")
    elif key == ord('-') or key == ord('_'):
        threshold = max(0.0, threshold - 0.05)
        print(f"Threshold decreased to: {threshold:.2f}")
    elif key == ord('s'):
        stabilization_enabled = not stabilization_enabled
        print(f"Stabilization: {'ON' if stabilization_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()