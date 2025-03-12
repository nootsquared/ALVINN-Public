import cv2
import numpy as np
import open3d as o3d

def generate_3d_map(raw_depth_map, raw_color_map, frame_index, depth_scale=1.0):
    def depth_to_point_cloud(depth_map, color_map, depth_scale):
        h, w = depth_map.shape
        points = []
        colors = []
        for v in range(h):
            for u in range(w):
                z = depth_map[v, u] * depth_scale
                if z == 0:
                    continue
                x = u
                y = v
                points.append([x, y, z])
                colors.append(color_map[v, u] / 255.0)
        
        return np.array(points), np.array(colors)

    def save_point_cloud(depth_map, color_map, frame_index, depth_scale):
        points, colors = depth_to_point_cloud(depth_map, color_map, depth_scale)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        filename = f"frame_{frame_index}.ply"
        o3d.io.write_point_cloud(filename, point_cloud)

    save_point_cloud(raw_depth_map, raw_color_map, frame_index, depth_scale)
