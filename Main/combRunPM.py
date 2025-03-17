import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
import time
import torch
import open3d as o3d
from datetime import datetime
import argparse

# Import saliency and depth models
from Saliency_Predictions.model import fastSal as fastsal
from Saliency_Predictions.utils import load_weight
from Saliency_Predictions.fastSalGCComb import process_frame_saliency
from Metric_Depth_Estimation.metric_depth.alvinn_depth.dpt import ALVINNDepth

def create_point_cloud(depth_map, color_frame, mask=None, contours=None, down_sample=True):
    """
    Create a point cloud from a depth map and color frame.
    
    Args:
        depth_map: Normalized depth map
        color_frame: Original color frame
        mask: Binary saliency mask (optional)
        contours: List of contours (optional)
        down_sample: If True, downsample the point cloud for better performance
    
    Returns:
        o3d.geometry.PointCloud: Point cloud object that can be saved or visualized
    """
    h, w = depth_map.shape
    
    # Create a regular grid of coordinates
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert depth to Z coordinates (scaled for better visualization)
    # Invert depth so closer objects appear closer to camera
    z_coords = (1.0 - depth_map) * 10
    
    # Create 3D coordinates - shift to center camera at origin
    points = np.stack([x_grid - w/2, y_grid - h/2, z_coords], axis=-1)
    
    # Reshape to a list of points
    points = points.reshape(-1, 3)
    
    # Prepare colors - highlighting salient regions if mask is provided
    if mask is not None and contours is not None:
        # Create a copy of the color image for highlighting
        highlight_img = color_frame.copy()
        
        # Draw contours
        cv2.drawContours(highlight_img, contours, -1, (0, 255, 0), 2)
        
        # Fill contours with semi-transparent highlight
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            
            # Create a mask for this contour
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 1, thickness=-1)
            
            # Highlight the region
            highlight_img[contour_mask > 0] = highlight_img[contour_mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        
        # Use highlighted colors
        colors = highlight_img.reshape(-1, 3) / 255.0
    else:
        # Use original frame colors
        colors = color_frame.reshape(-1, 3) / 255.0
    
    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Remove invalid points (e.g. where depth is 0)
    valid_indices = np.where(z_coords.reshape(-1) > 0.01)[0]
    pcd = pcd.select_by_index(valid_indices)
    
    # Optionally downsample the point cloud for better performance
    if down_sample:
        pcd = pcd.voxel_down_sample(voxel_size=1.0)  # adjust voxel_size as needed
    
    return pcd

def record_point_clouds(output_dir="point_cloud_data", max_frames=100, threshold=0.3):
    """
    Record a sequence of point clouds from webcam input.
    
    Args:
        output_dir: Directory to save point clouds
        max_frames: Maximum number of frames to record
        threshold: Initial saliency threshold
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    
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
    depth_weight_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'Metric_Depth_Estimation', 'checkpoints', 'depth_indoor.pth')
    depth_model = ALVINNDepth(
        encoder='vits',  # Try small ViT for better performance
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=80
    )
    depth_model.load_state_dict(torch.load(depth_weight_path, map_location='cpu'))
    depth_model.eval()
    DEVICE = 'cuda' if gpu else 'cpu'
    depth_model = depth_model.to(DEVICE)
    depth_input_size = 518  # as used in the depth code

    # ---- Start Webcam ----
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    frames_processed = 0
    recording = False
    
    print("\nControls:")
    print("  'r' - Start/stop recording")
    print("  '+'/'-' - Adjust saliency threshold")
    print("  'q' - Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        original_frame = frame.copy()
        
        # Process saliency for the current frame
        mask, contours, colored_sal = process_frame_saliency(saliency_model, frame, threshold, gpu, probability_output=False)
        
        # Compute depth map
        with torch.no_grad():
            depth = depth_model.infer_image(frame, depth_input_size)
        
        # Normalize depth for visualization and calculations
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # Create basic overlay for preview
        overlay_frame = cv2.addWeighted(frame, 0.75, colored_sal, 0.25, 0)
        
        # Process contours and show depth values
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            
            # Create mask for contour
            contour_mask = np.zeros(depth_norm.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 1, thickness=-1)
            
            # Compute average depth
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
                
            # Overlay text showing average depth
            cv2.putText(overlay_frame, f"{avg_depth:.2f}", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw the contour
            cv2.drawContours(overlay_frame, [cnt], -1, (0, 0, 255), 2)
        
        # If recording, save point cloud
        if recording and frames_processed < max_frames:
            # Create and save point cloud
            pcd = create_point_cloud(depth_norm, original_frame, mask, contours)
            pcd_path = os.path.join(output_path, f"frame_{frames_processed:04d}.ply")
            o3d.io.write_point_cloud(pcd_path, pcd)
            
            # Save original image for reference
            img_path = os.path.join(output_path, f"frame_{frames_processed:04d}.jpg")
            cv2.imwrite(img_path, original_frame)
            
            frames_processed += 1
            recording_text = f"Recording: {frames_processed}/{max_frames}"
            
            # Create manifest file with metadata
            with open(os.path.join(output_path, "manifest.txt"), 'w') as f:
                f.write(f"Recording date: {timestamp}\n")
                f.write(f"Frames: {frames_processed}\n")
                f.write(f"Saliency threshold: {threshold}\n")
        else:
            recording_text = "Press 'r' to record"
            if frames_processed >= max_frames:
                recording = False
                print(f"Finished recording {max_frames} frames.")
                
                # Create a combined point cloud of all frames
                combined_pcd = o3d.geometry.PointCloud()
                for i in range(frames_processed):
                    pcd_path = os.path.join(output_path, f"frame_{i:04d}.ply")
                    if os.path.exists(pcd_path):
                        frame_pcd = o3d.io.read_point_cloud(pcd_path)
                        combined_pcd += frame_pcd
                
                # Optionally downsample the combined point cloud
                combined_pcd = combined_pcd.voxel_down_sample(voxel_size=1.0)
                
                # Save the combined point cloud
                combined_path = os.path.join(output_path, "combined.ply")
                o3d.io.write_point_cloud(combined_path, combined_pcd)
                
                # Reset for another recording if needed
                frames_processed = 0
        
        # Display FPS and recording status
        end_time = time.time()
        fps_measured = 1 / (end_time - start_time)
        cv2.putText(overlay_frame, f"FPS: {fps_measured:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay_frame, recording_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay_frame, f"Threshold: {threshold:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show the preview
        cv2.imshow("Combined Saliency & Depth", overlay_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            if recording and frames_processed >= max_frames:
                # Reset if starting a new recording
                frames_processed = 0
        elif key == ord('+') or key == ord('='):
            threshold = min(1.0, threshold + 0.05)
            print(f"Threshold increased to: {threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            threshold = max(0.0, threshold - 0.05)
            print(f"Threshold decreased to: {threshold:.2f}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"Point clouds saved to: {output_path}")
    return output_path

def view_point_cloud(path):
    """
    Open an interactive viewer for a point cloud.
    
    Args:
        path: Path to a .ply point cloud file or directory containing point clouds
    """
    if os.path.isdir(path):
        # If a directory is provided, look for a combined point cloud or the first frame
        combined_path = os.path.join(path, "combined.ply")
        if os.path.exists(combined_path):
            pcd_path = combined_path
            print(f"Loading combined point cloud from {pcd_path}")
        else:
            # Find the first frame
            for file in sorted(os.listdir(path)):
                if file.endswith(".ply") and file.startswith("frame_"):
                    pcd_path = os.path.join(path, file)
                    print(f"Loading point cloud from {pcd_path}")
                    break
            else:
                print(f"No point cloud files found in {path}")
                return
    else:
        # Direct path to a .ply file
        if not path.endswith(".ply"):
            print("File must be a .ply point cloud")
            return
        pcd_path = path
    
    # Load the point cloud
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        print(f"Point cloud contains {len(pcd.points)} points")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # Create a visualization window
    print("Starting interactive viewer. Press 'q' to exit.")
    
    # Set up visualization window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Point Cloud Viewer", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set default viewpoint for better initial view
    view_control = vis.get_view_control()
    
    # Improve visualization by setting a reasonably sized bounding box
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    size = max(extent)
    
    # Set camera parameters for a good initial view
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])  # Look along negative Z axis
    view_control.set_up([0, -1, 0])     # Up is negative Y (screen coordinates)
    
    # Add helpful information
    print("\nViewer controls:")
    print("  Left mouse: Rotate")
    print("  Right mouse: Pan")
    print("  Mouse wheel: Zoom")
    print("  'r': Reset view")
    print("  'c': Change background color")
    print("  'q': Close viewer")
    
    # Register the key callbacks
    def close_callback(vis):
        return False
    
    def reset_view(vis):
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        return False
    
    def change_background(vis):
        opt = vis.get_render_option()
        if np.sum(opt.background_color) > 1.5:  # If light background
            opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark
        else:
            opt.background_color = np.array([0.9, 0.9, 0.9])  # Light
        return False
    
    vis.register_key_callback(ord("Q"), close_callback)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("C"), change_background)
    
    # Show points and start interactive loop
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background by default
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Record and view 3D point clouds from webcam.")
    parser.add_argument("--mode", type=str, default="record", choices=["record", "view"],
                        help="Mode: 'record' to capture new data, 'view' to visualize existing data")
    parser.add_argument("--path", type=str, default=None,
                        help="For view mode: path to point cloud file (.ply) or directory containing point clouds")
    parser.add_argument("--output", type=str, default="point_cloud_data",
                        help="For record mode: directory to save point cloud data")
    parser.add_argument("--frames", type=int, default=100,
                        help="For record mode: maximum number of frames to record")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="For record mode: initial saliency threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    if args.mode == "record":
        output_path = record_point_clouds(args.output, args.frames, args.threshold)
        
        # Automatically switch to view mode if recording was successful
        if output_path and os.path.exists(output_path):
            print(f"Switching to viewer mode for {output_path}")
            view_point_cloud(output_path)
    
    elif args.mode == "view":
        if not args.path:
            # If no path provided, find the most recent recording
            if os.path.exists("point_cloud_data"):
                dirs = sorted([os.path.join("point_cloud_data", d) for d in os.listdir("point_cloud_data") 
                              if os.path.isdir(os.path.join("point_cloud_data", d))])
                if dirs:
                    latest_dir = dirs[-1]  # Get the most recent directory
                    print(f"No path specified, using most recent recording: {latest_dir}")
                    view_point_cloud(latest_dir)
                else:
                    print("No recordings found. Run in 'record' mode first.")
            else:
                print("No recordings found. Run in 'record' mode first.")
        else:
            view_point_cloud(args.path)

if __name__ == "__main__":
    main()
