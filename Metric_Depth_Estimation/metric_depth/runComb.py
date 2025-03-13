import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch
import time
import glob

from alvinn_depth.dpt import ALVINNDepth
from pointmap import generate_3d_map
from groundplane import remove_ground

def run_webcam(depth_model, input_size=518, grayscale=False, cam_id=0):
    """Run depth estimation on webcam feed with temporal stability."""
    # Set up webcam
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    print("Press 'q' to quit, 's' to toggle stabilization")
    
    # For FPS calculation
    prev_frame_time = 0
    
    # For temporal stability
    prev_depth = None
    prev_frame = None
    stabilization_enabled = True
    
    # Parameters for stabilization
    consistency_threshold = 0.1  # Threshold for depth value consistency (smaller = more stable)
    max_stabilization = 0.7      # Maximum stabilization to apply (higher values reduce flickering more)
    motion_threshold = 30.0      # Threshold for motion detection (lower = more sensitive)
    
    # For motion detection
    movement_detection_enabled = True
    high_movement_threshold = 0.4  # Above this percentage of changed pixels = high movement
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam.")
            break
        
        # Start time for FPS calculation
        start_time = time.time()
        
        # Process frame to get depth map
        current_depth = depth_model.infer_image(frame, input_size)
        
        # Detect motion using frame differences if previous frame exists
        high_movement = False
        movement_percentage = 0
        
        if movement_detection_enabled and prev_frame is not None:
            # Convert frames to grayscale
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference between frames
            frame_diff = cv2.absdiff(current_gray, prev_gray)
            
            # Apply threshold to identify significant changes
            _, motion_mask = cv2.threshold(frame_diff, motion_threshold, 1, cv2.THRESH_BINARY)
            
            # Calculate percentage of pixels with motion
            movement_percentage = np.mean(motion_mask) * 100
            
            # Determine if we have high movement
            high_movement = (movement_percentage > high_movement_threshold * 100)
        
        # Store current frame for next iteration's motion detection
        prev_frame = frame.copy()
        
        # Apply depth stabilization if enabled and we don't have high movement
        if stabilization_enabled and prev_depth is not None and not high_movement:
            # Calculate absolute difference between current and previous depth maps
            depth_diff = np.abs(current_depth - prev_depth)
            
            # Normalize the difference relative to the depth values
            relative_diff = depth_diff / (np.abs(current_depth) + 1e-6)
            
            # Create a mask for inconsistent values (where the relative difference exceeds threshold)
            # These are the pixels that are likely flickering
            inconsistency_mask = (relative_diff > consistency_threshold).astype(np.float32)
            
            # Apply spatial smoothing to the mask to avoid sharp transitions
            inconsistency_mask = cv2.GaussianBlur(inconsistency_mask, (15, 15), 0)
            
            # Stabilize only the inconsistent areas, with more stabilization for more inconsistent areas
            stabilization_weight = np.clip(inconsistency_mask * max_stabilization, 0, max_stabilization)
            
            # Blend current depth with previous depth based on stabilization weight
            stabilized_depth = (1 - stabilization_weight) * current_depth + stabilization_weight * prev_depth
            
            # Calculate global inconsistency percentage for display
            inconsistency_percentage = np.mean(inconsistency_mask) * 100
            
            # For smooth transition when coming out of high movement
            if prev_depth is not None:
                # Apply a small amount of global stabilization to prevent sudden changes
                stabilized_depth = 0.95 * stabilized_depth + 0.05 * current_depth
        else:
            stabilized_depth = current_depth
            inconsistency_percentage = 0
        
        # Update previous depth for next iteration's comparison
        prev_depth = current_depth.copy()  
        
        # Normalize depth for visualization
        depth_norm = (stabilized_depth - stabilized_depth.min()) / (stabilized_depth.max() - stabilized_depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        
        # Apply colormap to depth
        if grayscale:
            depth_display = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
        else:
            depth_display = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        # Display FPS and stabilization status
        cv2.putText(depth_display, f"FPS: {fps:.1f}", 
                   (depth_display.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        if stabilization_enabled:
            status_text = f"Stab: ON"
            if high_movement:
                status_text += f" (PAUSED - {movement_percentage:.1f}% motion)"
            else:
                status_text += f" ({inconsistency_percentage:.1f}% unstable)"
                
            cv2.putText(depth_display, status_text, 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2)
        else:
            cv2.putText(depth_display, "Stab: OFF", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
        
        # Display the depth map
        cv2.imshow("Depth Map", depth_display)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            stabilization_enabled = not stabilization_enabled
            print(f"Stabilization: {'ON' if stabilization_enabled else 'OFF'}")
        elif key == ord('m'):
            movement_detection_enabled = not movement_detection_enabled
            print(f"Motion Detection: {'ON' if movement_detection_enabled else 'OFF'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metric Depth Module within ALVINN')
    
    parser.add_argument('--img-path', type=str, default='media/input_video_4.mp4')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='depth/vid_depth_outputs')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    # Fix the default path to use the absolute path
    default_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'checkpoints', 'depth_indoor.pth')
    parser.add_argument('--load-from', type=str, default=default_model_path)
    parser.add_argument('--max-depth', type=float, default=80)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--webcam', dest='webcam', action='store_true', help='run live webcam with depth overlay')
    parser.add_argument('--cam-id', type=int, default=0, help='webcam device index')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = ALVINNDepth(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu', weights_only=False))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # If webcam flag is set, run the webcam livestream
    if args.webcam:
        run_webcam(depth_anything, args.input_size, args.grayscale, args.cam_id)
    # Otherwise run the regular video processing
    else:
        if os.path.isfile(args.img_path):
            if args.img_path.endswith('txt'):
                with open(args.img_path, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [args.img_path]
        else:
            filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        
        os.makedirs(args.outdir, exist_ok=True)
        
        # Process each file independently
        for k, filename in enumerate(filenames):
            print(f'Processing {k+1}/{len(filenames)}: {filename}')
            
            # For video files
            if filename.endswith(('.mp4', '.avi', '.mov')):
                cap = cv2.VideoCapture(filename)
                
                # Get video details
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create output video writer
                output_path = os.path.join(args.outdir, os.path.basename(filename))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame to get depth map
                    depth = depth_model.infer_image(frame, args.input_size)
                    
                    # Normalize depth for visualization
                    depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    depth_norm = depth_norm.astype(np.uint8)
                    
                    # Apply colormap to depth
                    if args.grayscale:
                        depth_display = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
                    else:
                        depth_display = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    
                    # Resize if needed
                    if depth_display.shape[:2] != (height, width):
                        depth_display = cv2.resize(depth_display, (width, height))
                    
                    # Write frame to output video
                    out.write(depth_display)
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Processed {frame_count} frames")
                
                cap.release()
                out.release()
                
            # For image files
            elif filename.endswith(('.jpg', '.jpeg', '.png')):
                image = cv2.imread(filename)
                
                # Process image to get depth map
                depth = depth_model.infer_image(image, args.input_size)
                
                # Normalize depth for visualization
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth_norm = depth_norm.astype(np.uint8)
                
                # Apply colormap to depth
                if args.grayscale:
                    depth_display = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
                else:
                    depth_display = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                
                # Save depth map
                output_path = os.path.join(args.outdir, os.path.basename(filename))
                cv2.imwrite(output_path, depth_display)
                
                if args.save_numpy:
                    np.save(output_path.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy'), depth)