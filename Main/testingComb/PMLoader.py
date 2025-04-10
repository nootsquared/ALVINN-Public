import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2
import numpy as np
import time
import open3d as o3d
import argparse
from glob import glob

class PointCloudPlayer:
    """
    Interactive player for navigating through a sequence of 3D point cloud frames.
    Provides controls for playback with enhanced depth visualization.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ply_files = sorted(glob(os.path.join(data_dir, "frame_*.ply")))
        self.frame_count = len(self.ply_files)
        
        if self.frame_count == 0:
            raise ValueError(f"No point cloud frames found in {data_dir}")
            
        self.current_frame = 0
        self.playing = False
        self.play_speed = 5  # frames per second
        self.depth_scale = 50.0  # Fixed depth scale of 10x
        self.last_time = time.time()
        self.original_pcd = None  # Store original point cloud for scaling
        
        # Initialize Open3D visualization
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Point Cloud Video Player", width=1280, height=720)
        
        # Load the first frame
        self.pcd = self.load_point_cloud(self.ply_files[self.current_frame])
        self.vis.add_geometry(self.pcd)
        
        # Configure view
        self.setup_view()
        self.setup_callbacks()
        
        # Set initial render options
        opt = self.vis.get_render_option()
        opt.point_size = 3.0  # Larger points for better visibility
        opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background

    def load_point_cloud(self, filepath):
        """Load a point cloud from file with fixed depth scale applied"""
        try:
            # Load original point cloud without any scaling
            self.original_pcd = o3d.io.read_point_cloud(filepath)
            
            # Create a copy for display with depth scaling applied
            pcd = o3d.geometry.PointCloud()
            points = np.asarray(self.original_pcd.points)
            colors = np.asarray(self.original_pcd.colors)
            
            # Apply depth scaling to Z coordinates
            scaled_points = points.copy()
            scaled_points[:, 2] = points[:, 2] * self.depth_scale
            
            pcd.points = o3d.utility.Vector3dVector(scaled_points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            return pcd
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def setup_view(self):
        """Configure initial camera viewpoint"""
        view_control = self.vis.get_view_control()
        
        # Set a reasonable default view
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])  # Look along negative Z axis
        view_control.set_up([0, -1, 0])     # Up is negative Y
    
    def setup_callbacks(self):
        """Register all keyboard and mouse callbacks"""
        # Navigation controls
        self.vis.register_key_callback(ord(" "), self.toggle_playback)
        self.vis.register_key_callback(ord("."), self.next_frame)
        self.vis.register_key_callback(ord(","), self.prev_frame)
        
        # Frame jumping
        self.vis.register_key_callback(ord("]"), self.jump_forward)
        self.vis.register_key_callback(ord("["), self.jump_backward)
        
        # Speed control
        self.vis.register_key_callback(ord("+"), self.increase_speed)
        self.vis.register_key_callback(ord("-"), self.decrease_speed)
        
        # View controls
        self.vis.register_key_callback(ord("r"), self.reset_view)
        self.vis.register_key_callback(ord("c"), self.change_background)
        self.vis.register_key_callback(ord("p"), self.increase_point_size)
        self.vis.register_key_callback(ord("o"), self.decrease_point_size)
        
        # Exit
        self.vis.register_key_callback(ord("q"), self.close)
    
    def toggle_playback(self, vis):
        """Toggle between play and pause"""
        self.playing = not self.playing
        print("Playback: " + ("Playing" if self.playing else "Paused"))
        return False
    
    def next_frame(self, vis):
        """Go to next frame"""
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
            self.update_frame()
        return False
    
    def prev_frame(self, vis):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()
        return False
    
    def jump_forward(self, vis):
        """Jump 10 frames forward"""
        self.current_frame = min(self.current_frame + 10, self.frame_count - 1)
        self.update_frame()
        return False
    
    def jump_backward(self, vis):
        """Jump 10 frames backward"""
        self.current_frame = max(self.current_frame - 10, 0)
        self.update_frame()
        return False
    
    def increase_speed(self, vis):
        """Increase playback speed"""
        self.play_speed = min(30, self.play_speed + 1)
        print(f"Playback speed: {self.play_speed} fps")
        return False
    
    def decrease_speed(self, vis):
        """Decrease playback speed"""
        self.play_speed = max(1, self.play_speed - 1)
        print(f"Playback speed: {self.play_speed} fps")
        return False
    
    def increase_point_size(self, vis):
        """Increase point size"""
        opt = self.vis.get_render_option()
        opt.point_size = min(10.0, opt.point_size + 0.5)
        print(f"Point size: {opt.point_size:.1f}")
        return False
    
    def decrease_point_size(self, vis):
        """Decrease point size"""
        opt = self.vis.get_render_option()
        opt.point_size = max(1.0, opt.point_size - 0.5)
        print(f"Point size: {opt.point_size:.1f}")
        return False
    
    def reset_view(self, vis):
        """Reset camera view"""
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        return False
    
    def change_background(self, vis):
        """Toggle background color between dark and light"""
        opt = self.vis.get_render_option()
        if np.sum(opt.background_color) > 1.5:  # If light background
            opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark
            print("Background: Dark")
        else:
            opt.background_color = np.array([0.9, 0.9, 0.9])  # Light
            print("Background: Light")
        return False
    
    def close(self, vis):
        """Close the visualization"""
        return False
    
    def update_frame(self, force_update=False):
        """Update the displayed point cloud to current frame"""
        if 0 <= self.current_frame < self.frame_count:
            # Load the current frame
            new_pcd = self.load_point_cloud(self.ply_files[self.current_frame])
            if new_pcd:
                # Update points and colors
                self.pcd.points = new_pcd.points
                self.pcd.colors = new_pcd.colors
                
                # Update the geometry in the visualizer
                self.vis.update_geometry(self.pcd)
                
                # Print current frame info
                print(f"Frame: {self.current_frame+1}/{self.frame_count}")
    
    def run(self):
        """Main run loop for the player"""
        print("\nPoint Cloud Video Player Controls:")
        print("  Space - Play/pause")
        print("  ,/. - Previous/next frame")
        print("  [/] - Jump 10 frames backward/forward")
        print("  +/- - Increase/decrease playback speed")
        print("\nView Controls:")
        print("  r - Reset view")
        print("  c - Toggle background color")
        print("  p/o - Increase/decrease point size")
        print("  q - Quit")
        print("\nMouse controls:")
        print("  Left mouse - Rotate")
        print("  Right mouse - Pan")
        print("  Mouse wheel - Zoom")
        
        print(f"\nDepth scale: {self.depth_scale:.1f}x (fixed)")
        
        # Main visualization loop
        while True:
            # Handle automatic playback
            if self.playing:
                current_time = time.time()
                if current_time - self.last_time > 1.0 / self.play_speed:
                    self.last_time = current_time
                    
                    # Advance frame
                    self.current_frame += 1
                    if self.current_frame >= self.frame_count:
                        self.current_frame = 0  # Loop back to start
                    
                    self.update_frame()
            
            # Update visualization
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
        
        self.vis.destroy_window()

def show_frame_info(data_dir):
    """Show information about the point cloud sequence"""
    ply_files = sorted(glob(os.path.join(data_dir, "frame_*.ply")))
    frame_count = len(ply_files)
    
    if frame_count == 0:
        print(f"No point cloud frames found in {data_dir}")
        return
    
    # Load and analyze the first frame
    try:
        pcd = o3d.io.read_point_cloud(ply_files[0])
        points = np.asarray(pcd.points)
        
        # Extract depth statistics
        z_values = points[:, 2]
        depth_min = np.min(z_values)
        depth_max = np.max(z_values)
        depth_avg = np.mean(z_values)
        depth_range = depth_max - depth_min
        
        print(f"Point Cloud Sequence Information:")
        print(f"  Location: {data_dir}")
        print(f"  Total frames: {frame_count}")
        print(f"  Points per frame: {len(points)}")
        print(f"  Depth range: {depth_min:.3f} to {depth_max:.3f} (range: {depth_range:.3f})")
        print(f"  Using fixed 10x depth scale for enhanced visualization")
        
        # Read manifest if available
        manifest_path = os.path.join(data_dir, "manifest.txt")
        if os.path.exists(manifest_path):
            print("\nManifest data:")
            with open(manifest_path, 'r') as f:
                manifest_data = f.read()
                print(manifest_data)
    
    except Exception as e:
        print(f"Error analyzing point cloud data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Play recorded point cloud sequences as video")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to directory containing point cloud frames")
    parser.add_argument("--info", action="store_true",
                        help="Show information about the point cloud sequence without playing")
    
    args = parser.parse_args()
    
    # If no path specified, try to find the most recent recording
    data_dir = args.path
    if data_dir is None:
        if os.path.exists("point_cloud_data"):
            dirs = sorted([os.path.join("point_cloud_data", d) for d in os.listdir("point_cloud_data") 
                          if os.path.isdir(os.path.join("point_cloud_data", d))])
            if dirs:
                data_dir = dirs[-1]  # Get the most recent directory
                print(f"No path specified, using most recent recording: {data_dir}")
            else:
                print("No recordings found.")
                return
        else:
            print("No recordings found.")
            return
    
    # Show sequence information
    show_frame_info(data_dir)
    
    # Just show info if requested
    if args.info:
        return
    
    # Start the player with fixed 10x depth scale
    try:
        player = PointCloudPlayer(data_dir)
        player.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()