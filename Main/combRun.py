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

# ---- Start Webcam Loop ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    exit(1)

threshold = 0.3  # initial saliency threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()
    
    # Process saliency for the current frame
    mask, contours, colored_sal = process_frame_saliency(saliency_model, frame, threshold, gpu, probability_output=False)
    
    # Compute depth map for the same frame
    # Using the depth model's infer_image method as in runComb (assumes it processes a single frame)
    # Preprocess frame for depth model (using the model's own preprocessing inside infer_image)
    with torch.no_grad():
        # The depth inference method; assume infer_image exists
        depth = depth_model.infer_image(frame, depth_input_size)
    # Normalize depth for visualization and calculations
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    
    # Prepare overlay: blend the colored saliency output with the original frame
    overlay_frame = cv2.addWeighted(frame, 0.75, colored_sal, 0.25, 0)
    
    # For each contour, compute average depth within that blob
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue  # ignore tiny blobs
        # Create a blank mask for the contour
        contour_mask = np.zeros(depth_norm.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 1, thickness=-1)
        # Compute average depth; depth_norm is float in [0,1], scale if desired
        if np.sum(contour_mask) > 0:
            avg_depth = np.sum(depth_norm * contour_mask) / np.sum(contour_mask)
        else:
            avg_depth = 0
        # Get contour centroid
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = cnt[0][0]
        # Overlay text showing average depth (you can scale avg_depth as needed)
        cv2.putText(overlay_frame, f"{avg_depth:.2f}", (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        # Optionally, draw the contour
        cv2.drawContours(overlay_frame, [cnt], -1, (0,0,255), 2)
    
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

cap.release()
cv2.destroyAllWindows()