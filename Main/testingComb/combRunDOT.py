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

# Import depth model
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
depth_weight_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Metric_Depth_Estimation',
    'checkpoints',
    'depth_indoor.pth'
)
depth_model = ALVINNDepth(
    encoder='vits',
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=80
)
depth_model.load_state_dict(torch.load(depth_weight_path, map_location='cpu'))
depth_model.eval()
DEVICE = 'cuda' if gpu else 'cpu'
depth_model = depth_model.to(DEVICE)
depth_input_size = 518  # As used in the depth code

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

threshold = 0.5  # Initial saliency threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Get raw saliency prediction
    mask, _, colored_sal = process_frame_saliency(
        saliency_model, frame, threshold, gpu, probability_output=False
    )
    # 'mask' is a binary mask. We won't use contours.

    # Compute depth
    with torch.no_grad():
        depth = depth_model.infer_image(frame, depth_input_size)

    # Apply temporal stabilization if enabled
    if stabilization_enabled and prev_depth is not None and prev_frame is not None:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion = np.mean(np.abs(frame_gray.astype(float) - prev_gray.astype(float)))
        if motion < motion_threshold:
            stability_weight = max_stabilization * (1 - motion / motion_threshold)
            depth = (1 - stability_weight) * depth + stability_weight * prev_depth

    # Store current frame/depth
    prev_frame = frame.copy()
    prev_depth = depth.copy()

    # Convert to metric depth
    depth_meters = depth * depth_model.max_depth / 100

    # Prepare overlay: blend the colored saliency output with the original frame
    overlay_frame = cv2.addWeighted(frame, 0.75, colored_sal, 0.25, 0)

    # Convert saliency to grayscale
    sal_gray = cv2.cvtColor(colored_sal, cv2.COLOR_BGR2GRAY)
    # Threshold saliency again (using the same threshold but scaled to 0-255)
    _, bin_sal = cv2.threshold(sal_gray, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Find connected components of high-saliency pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_sal, connectivity=8)
    
    # Minimum area to consider for a valid “peak” region
    min_area = 100

    # For each connected component, find its saliency peak and mark depth
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # mask_label is the region of this label
        mask_label = (labels == label_id).astype(np.uint8)

        # Use minMaxLoc on the saliency region to find the highest-intensity pixel
        blob_sal = sal_gray * mask_label
        _, _, _, max_loc = cv2.minMaxLoc(blob_sal)
        peak_x, peak_y = max_loc  # (peak_x, peak_y)

        # Get the depth at that peak
        if 0 <= peak_y < depth_meters.shape[0] and 0 <= peak_x < depth_meters.shape[1]:
            peak_depth = depth_meters[peak_y, peak_x]
        else:
            peak_depth = 0

        # Draw a small circle for the peak
        cv2.circle(overlay_frame, (peak_x, peak_y), 5, (0, 255, 0), -1)
        # Print the depth
        cv2.putText(
            overlay_frame,
            f"{peak_depth:.2f}m",
            (peak_x + 5, peak_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Show stabilization status
    stab_status = "ON" if stabilization_enabled else "OFF"
    cv2.putText(overlay_frame, f"Stab: {stab_status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time) if (end_time - start_time) else 0
    cv2.putText(overlay_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Combined Saliency & Depth", overlay_frame)
    key = cv2.waitKey(1) & 0xFF

    # Key controls
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