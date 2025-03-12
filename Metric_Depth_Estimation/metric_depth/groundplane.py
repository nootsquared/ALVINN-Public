import numpy as np
import cv2

def remove_ground(depth_map):
    # Step 1: Normalize the depth map
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    
    # Step 2: Apply logarithmic transformation
    depth_map_log = np.log1p(depth_map_normalized)
    
    # Step 3: Use Sobel operator to detect edges
    sobel_x = cv2.Sobel(depth_map_log, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(depth_map_log, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobel_x, sobel_y)
    
    # Step 4: Threshold to identify ground
    height, width = sobel.shape
    bottom_quarter = height // 4
    
    # Apply different thresholds for the bottom 1/4 and the rest of the image
    ground_mask = np.zeros_like(sobel)
    _, ground_mask[:height - bottom_quarter, :] = cv2.threshold(sobel[:height - bottom_quarter, :], 0.05, 1, cv2.THRESH_BINARY)
    _, ground_mask[height - bottom_quarter:, :] = cv2.threshold(sobel[height - bottom_quarter:, :], 0.025, 1, cv2.THRESH_BINARY)
    
    # Step 5: Remove ground pixels
    depth_map_no_ground = depth_map.copy()
    depth_map_no_ground[ground_mask > 0] = 0
    
    # Step 6: Apply smoothing filter
    depth_map_no_ground_smoothed = cv2.GaussianBlur(depth_map_no_ground, (5, 5), 0)
    
    return depth_map_no_ground_smoothed

import numpy as np
import cv2

def logarithmic_threshold(sobel, start_thresh, end_thresh):
    height, width = sobel.shape
    mask = np.zeros_like(sobel)
    
    for i in range(height):

        log_scale = np.log1p(height - i) / np.log1p(height)
        threshold = start_thresh + (end_thresh - start_thresh) * log_scale

        _, row_mask = cv2.threshold(sobel[i, :], threshold, 1, cv2.THRESH_BINARY)
        mask[i, :] = row_mask.squeeze()
    
    return mask

def remove_ground(depth_map, start_thresh=0.005, end_thresh=0.05):

    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    depth_map_log = np.log1p(depth_map_normalized)

    sobel_x = cv2.Sobel(depth_map_log, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(depth_map_log, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobel_x, sobel_y)
    
    ground_mask = logarithmic_threshold(sobel, start_thresh, end_thresh)
    
    depth_map_no_ground = depth_map.copy()
    depth_map_no_ground[ground_mask > 0] = 0
    
    depth_map_no_ground_smoothed = cv2.GaussianBlur(depth_map_no_ground, (5, 5), 0)
    
    return depth_map_no_ground_smoothed


# Try and making the log function dynamic???
# 2 and 3 are not as good outputs as 1

# import numpy as np
# import cv2

# def estimate_tilt(depth_map):

#     avg_depth_per_row = np.mean(depth_map, axis=1)
#     rows = np.arange(len(avg_depth_per_row))
#     tilt_angle = np.polyfit(rows, avg_depth_per_row, 1)[0]
    
#     return tilt_angle

# def adjust_thresholds(tilt_angle, base_start_thresh, base_end_thresh):
#     scaling_factor = 4.0
#     start_thresh = base_start_thresh * (1 + scaling_factor * abs(tilt_angle))
#     end_thresh = base_end_thresh * (1 - scaling_factor * abs(tilt_angle))
    
#     start_thresh = max(0.001, min(start_thresh, 0.1))
#     end_thresh = max(0.001, min(end_thresh, 0.1))
    
#     return start_thresh, end_thresh

# def logarithmic_threshold(sobel, start_thresh, end_thresh):
#     height, width = sobel.shape
#     mask = np.zeros_like(sobel)
    
#     for i in range(height):
#         log_scale = np.log1p(height - i) / np.log1p(height)
#         threshold = start_thresh + (end_thresh - start_thresh) * log_scale
#         _, row_mask = cv2.threshold(sobel[i, :], threshold, 1, cv2.THRESH_BINARY)
#         mask[i, :] = row_mask.squeeze()
    
#     return mask

# def remove_ground(depth_map, base_start_thresh=0.05, base_end_thresh=0.05):
    
#     tilt_angle = estimate_tilt(depth_map)

#     print(f"Tilt Angle: {tilt_angle}")
    
#     start_thresh, end_thresh = adjust_thresholds(tilt_angle, base_start_thresh, base_end_thresh)
    
#     depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
#     depth_map_log = np.log1p(depth_map_normalized)
    
#     sobel_x = cv2.Sobel(depth_map_log, cv2.CV_64F, 1, 0, ksize=5)
#     sobel_y = cv2.Sobel(depth_map_log, cv2.CV_64F, 0, 1, ksize=5)
#     sobel = np.hypot(sobel_x, sobel_y)
    
#     ground_mask = logarithmic_threshold(sobel, start_thresh, end_thresh)
    
#     depth_map_no_ground = depth_map.copy()
#     depth_map_no_ground[ground_mask > 0] = 0
    
#     depth_map_no_ground_smoothed = cv2.GaussianBlur(depth_map_no_ground, (5, 5), 0)
    
#     return depth_map_no_ground_smoothed