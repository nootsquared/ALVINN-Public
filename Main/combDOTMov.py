import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
import time
import torch
import os
import argparse

# Import saliency model and helper from fastSalGCComb
from Saliency_Predictions.model import fastSal as fastsal
from Saliency_Predictions.utils import load_weight
from Saliency_Predictions.fastSalGCComb import process_frame_saliency

# Import depth model
from Metric_Depth_Estimation.metric_depth.alvinn_depth.dpt import ALVINNDepth

def draw_motion_minimap(frame, motion_x, motion_y, box_size=80):
    """Draw a minimap showing camera motion direction with an arrow"""
    # Position in top-right corner with padding
    x_pos = frame.shape[1] - box_size - 20
    y_pos = 20
    
    # Create the minimap box
    cv2.rectangle(frame, (x_pos, y_pos), (x_pos + box_size, y_pos + box_size), (255, 255, 255), 1)
    
    # Center point of the box
    center_x = x_pos + box_size // 2
    center_y = y_pos + box_size // 2
    
    # Draw center point
    cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
    
    # Calculate motion magnitude for arrow length
    motion_magnitude = np.sqrt(motion_x**2 + motion_y**2)
    
    # Apply non-linear scaling to reduce sensitivity for small movements
    # and better represent large movements
    sensitivity = 1.5  # INCREASED from 0.5 to make movements more pronounced
    max_motion = 10.0  # REDUCED from 15.0 to make arrow reach max length sooner
    
    # Apply sensitivity adjustment and cap maximum motion
    adjusted_magnitude = min(max_motion, motion_magnitude * sensitivity)
    
    # Scale factors to control arrow length
    min_length = 5.0   # Minimum arrow length for visible motion
    max_length = box_size / 2 - 5  # Maximum arrow length (nearly to edge)
    
    # Calculate the normalized arrow length
    if motion_magnitude > 0.8:  # REDUCED from 2.0 to show arrows for smaller movements
        # Scale between min and max length based on adjusted magnitude
        arrow_length = min_length + (adjusted_magnitude / max_motion) * (max_length - min_length)
    else:
        arrow_length = 0  # No visible arrow for tiny movements
    
    if arrow_length > 0:
        # Normalize the direction vector
        if motion_magnitude > 0:
            norm_x = motion_x / motion_magnitude
            norm_y = motion_y / motion_magnitude
        else:
            norm_x, norm_y = 0, 0
            
        # Calculate arrow endpoint using normalized direction and scaled length
        # Invert x and y to match camera motion (when camera moves right, scene moves left)
        end_x = center_x - int(norm_x * arrow_length)
        end_y = center_y - int(norm_y * arrow_length)
        
        # Draw the arrow with color based on intensity
        # Green for slow, yellow for medium, red for fast
        if adjusted_magnitude < max_motion * 0.3:
            arrow_color = (0, 255, 0)  # Green
        elif adjusted_magnitude < max_motion * 0.7:
            arrow_color = (0, 255, 255)  # Yellow
        else:
            arrow_color = (0, 0, 255)  # Red
            
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                        arrow_color, 2, tipLength=0.3)
        
        # Add direction label
        direction = ""
        if abs(motion_y) > 0.8:  # REDUCED from 2.0 to show directions for smaller movements
            direction += "N" if motion_y < 0 else "S"
        if abs(motion_x) > 0.8:  # REDUCED from 2.0 to show directions for smaller movements
            direction += "E" if motion_x < 0 else "W"
        
        if direction:
            cv2.putText(frame, direction, (center_x - 10, y_pos + box_size + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Add intensity indicator
        intensity = f"{motion_magnitude:.1f}"
        cv2.putText(frame, intensity, (x_pos + 5, y_pos + box_size + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def process_video(input_path, output_path, threshold=0.5, stabilization_enabled=True):
    # ---- Load Models ----
    # Saliency Model
    print("Loading saliency model...")
    saliency_model = fastsal.fastsal(pretrain_mode=False, model_type='A')
    sal_weights = load_weight('Saliency_Predictions/weights/SALICON_A.pth', remove_decoder=False)[0]
    saliency_model.load_state_dict(sal_weights)
    saliency_model.eval()
    gpu = torch.cuda.is_available()
    if gpu:
        saliency_model.cuda()

    # Depth Model
    print("Loading depth model...")
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
    prev_gray = None
    consistency_threshold = 0.1
    max_stabilization = 0.7
    motion_threshold = 30.0
    
    # Variables for optical flow
    motion_x = 0
    motion_y = 0
    motion_smoothing = 0.8  # Increased for smoother transitions between motion updates

    # ---- Open input video ----
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video at {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30, (width, height))

    frame_count = 0
    processing_start = time.time()
    
    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        frame_count += 1
        
        if frame_count % 10 == 0:
            percent_done = (frame_count / total_frames) * 100
            print(f"Processing: {frame_count}/{total_frames} frames ({percent_done:.1f}%)")

        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow if we have a previous frame
        if prev_gray is not None:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Extract x and y components of the flow
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            # Calculate average flow (camera motion direction)
            # Ignore small movements to reduce noise
            mask = (np.abs(flow_x) > 0.7) | (np.abs(flow_y) > 0.7)  # Increased threshold for noise reduction
            if np.sum(mask) > 0:
                avg_x = np.mean(flow_x[mask])
                avg_y = np.mean(flow_y[mask])
                
                # Apply smoothing to the motion values
                motion_x = motion_smoothing * motion_x + (1 - motion_smoothing) * avg_x
                motion_y = motion_smoothing * motion_y + (1 - motion_smoothing) * avg_y
        
        # Update previous gray frame
        prev_gray = gray.copy()

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
            prev_gray_stabilize = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            motion = np.mean(np.abs(frame_gray.astype(float) - prev_gray_stabilize.astype(float)))
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
        
        # Minimum area to consider for a valid "peak" region
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
        processing_fps = 1 / (end_time - start_time) if (end_time - start_time) else 0
        cv2.putText(overlay_frame, f"FPS: {processing_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Add motion minimap to top right
        draw_motion_minimap(overlay_frame, motion_x, motion_y)

        # Write frame to output video
        out_vid.write(overlay_frame)

    # Clean up
    cap.release()
    out_vid.release()
    
    total_time = time.time() - processing_start
    print(f"Processing complete. {frame_count} frames processed in {total_time:.2f} seconds.")
    print(f"Average FPS: {frame_count / total_time:.2f}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with saliency and depth detection')
    parser.add_argument('--input', type=str, default="Media/Input_Videos/input_video_3.mp4",
                        help='Path to input video')
    parser.add_argument('--output', type=str, default="Media/Output_Videos/output_processed.mp4",
                        help='Path to output video')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Saliency threshold (0.0-1.0)')
    parser.add_argument('--no-stabilization', action='store_true',
                        help='Disable depth stabilization')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.threshold, not args.no_stabilization)

