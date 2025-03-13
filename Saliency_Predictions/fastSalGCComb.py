from Saliency_Predictions.model import fastSal as fastsal
import numpy as np
import argparse
import cv2
import time
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os

# Import only what's needed
from Saliency_Predictions.utils import load_weight

class video_dataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __getitem__(self, item):
        frame = self.frames[item]
        vgg_img, original_size = read_vgg_img(frame, (192, 256), from_array=True)
        return vgg_img, original_size, item

    def __len__(self):
        return len(self.frames)

def pytorch_normalize(tensor):
    # Define the normalization function
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

def read_vgg_img(frame, target_size, from_array=False):
    if from_array:
        vgg_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vgg_img = Image.fromarray(vgg_img)
    else:
        # Direct frame reading instead of using read_image
        vgg_img = cv2.imread(frame)
        vgg_img = cv2.cvtColor(vgg_img, cv2.COLOR_BGR2RGB)
        vgg_img = Image.fromarray(vgg_img)
    
    original_size = vgg_img.size
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if (target_size[0] != original_size[1] or target_size[1] != original_size[0]):
            # Using BICUBIC instead of LANCZOS
            vgg_img = vgg_img.resize((target_size[1], target_size[0]), Image.BICUBIC)
    vgg_img = np.asarray(vgg_img, dtype=np.float32)
    vgg_img = pytorch_normalize(torch.FloatTensor(vgg_img).permute(2, 0, 1) / 255.0)
    return vgg_img, np.asarray(original_size)

# ...existing code...

def post_process_png(prediction, original_size):
    """
    Process the prediction for PNG output format.
    
    Args:
        prediction: The model's prediction output
        original_size: Original image dimensions
        
    Returns:
        Processed prediction resized to original dimensions
    """
    # Ensure prediction values are between 0 and 1
    prediction = np.clip(prediction, 0, 1)
    
    # Resize to original dimension (width, height)
    prediction = cv2.resize(prediction, (original_size[0], original_size[1]))
    
    # Scale to 0-255 range
    prediction = prediction * 255
    
    return prediction

def post_process_probability2(prediction, original_size):
    """
    Process the prediction for probability output format.
    
    Args:
        prediction: The model's prediction output
        original_size: Original image dimensions
        
    Returns:
        Processed prediction as probability map resized to original dimensions
    """
    # Get min and max for normalization
    min_val = prediction.min()
    max_val = prediction.max()
    
    # Normalize the prediction to [0, 1] range if needed
    if max_val > min_val:
        prediction = (prediction - min_val) / (max_val - min_val)
    
    # Resize to original dimension (width, height)
    prediction = cv2.resize(prediction, (original_size[0], original_size[1]))
    
    return prediction

# ...existing code...

def process_frame_saliency(model, frame, threshold=0.3, gpu=True, probability_output=False):
    """
    Process a single frame using the saliency model.
    
    Returns:
        mask: binary numpy array from thresholded saliency map
        contours: list of detected contours
        colored_sal: saliency color map for visualization
    """
    import torch.nn as nn
    # Preprocess frame (using existing helper)
    vgg_img, original_size = read_vgg_img(frame, (192,256), from_array=True)
    x = vgg_img.unsqueeze(0)  # add batch dimension
    if gpu:
        x = x.float().cuda()
    y = model(x)
    if not probability_output:
        y = nn.Sigmoid()(y)
    if gpu:
        y = y.detach().cpu()
    y = y.numpy()
    # Single image prediction
    prediction = y[0,0,:,:]
    # Post-process saliency map
    if not probability_output:
        sal_map = post_process_png(prediction, original_size)
    else:
        sal_map = post_process_probability2(prediction, original_size)
    # Resize saliency map to match frame dimensions
    height, width = frame.shape[:2]
    sal_map = cv2.resize(sal_map, (width, height))
    sal_map_norm = cv2.normalize(sal_map, None, 0, 1, cv2.NORM_MINMAX)
    mask = (sal_map_norm > threshold).astype(np.uint8)
    # Find contours on the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a colormap for visualization
    colored_sal = cv2.applyColorMap((sal_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return mask, contours, colored_sal

def predict_webcam(model_type, finetune_dataset, probability_output, batch_size, gpu=True, initial_threshold=0.5):
    start_total_time = time.time()  # Start the timer
    threshold = initial_threshold  # Initialize threshold value

    model = fastsal.fastsal(pretrain_mode=False, model_type=model_type)
    state_dict, _ = load_weight('weights/{}_{}.pth'.format(finetune_dataset, model_type), remove_decoder=False)
    model.load_state_dict(state_dict)
    if gpu:
        model.cuda()

    cap = cv2.VideoCapture(0)  # Use webcam
    fps = 30  # Set a default FPS value
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames = [frame]
        video_data = video_dataset(frames)
        video_loader = DataLoader(video_data, batch_size=batch_size, shuffle=False, num_workers=0)

        for i, (x, original_size_list, _) in enumerate(video_loader):
            start_time = time.time()
            if gpu:
                x = x.float().cuda()
            y = model(x)
            if not probability_output: y = nn.Sigmoid()(y)
            if gpu:
                y = y.detach().cpu()
            y = y.numpy()
            for j, prediction in enumerate(y[:, 0, :, :]):
                original_size = original_size_list[j].numpy()
                
                if not probability_output:
                    saliency_map = post_process_png(prediction, original_size)
                else:
                    saliency_map = post_process_probability2(prediction, original_size)
                
                # Resize saliency map to match frame dimensions
                saliency_map = cv2.resize(saliency_map, (width, height))
                
                # Normalize to 0-1 range
                saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
                
                # Create a mask based on the threshold
                mask = (saliency_map > threshold).astype(np.uint8)
                
                # Apply colormap to saliency for visualization
                colored_saliency = cv2.applyColorMap(
                    (saliency_map * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                
                # Create overlay with saliency
                overlay_frame = cv2.addWeighted(frame, 0.75, colored_saliency, 0.25, 0)
                
                # Create binary mask first
                binary_mask = np.zeros_like(overlay_frame)
                # Only copy pixels from overlay_frame where mask is 1 (above threshold)
                binary_mask[mask > 0] = overlay_frame[mask > 0]
                
                # Find contours on the binary mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw red contours around the detected regions
                masked_output = binary_mask.copy()
                cv2.drawContours(masked_output, contours, -1, (0, 0, 255), 2)  # Red color (BGR format)
                
                # Calculate and display FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                
                # Add text with FPS and threshold information
                cv2.putText(masked_output, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(masked_output, f"Threshold: {threshold:.2f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(masked_output, "Press +/- to adjust threshold", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display the frame
                cv2.imshow('Masked Saliency', masked_output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('+') or key == ord('='):  # Increase threshold
                    threshold = min(1.0, threshold + 0.05)
                    print(f"Threshold increased to: {threshold:.2f}")
                elif key == ord('-') or key == ord('_'):  # Decrease threshold
                    threshold = max(0.0, threshold - 0.05)
                    print(f"Threshold decreased to: {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()

    end_total_time = time.time()  # End the timer
    total_time = end_total_time - start_total_time
    print(f"Total time taken: {total_time:.2f} seconds")

def predict_video(model_type, finetune_dataset, input_path, output_path,
                  probability_output, batch_size, gpu=True):
    start_total_time = time.time()  # Start the timer

    model = fastsal.fastsal(pretrain_mode=False, model_type=model_type)
    state_dict, _ = load_weight('weights/{}_{}.pth'.format(finetune_dataset, model_type), remove_decoder=False)
    model.load_state_dict(state_dict)
    if gpu:
        model.cuda()

    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    video_data = video_dataset(frames)
    video_loader = DataLoader(video_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Total frames: {frame_count}")

    # Use proper fourcc code
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID instead of mp4v
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    overlay_output_path = output_path.replace('.mp4', '_overlay.mp4')
    out_overlay = cv2.VideoWriter(overlay_output_path, fourcc, fps, (width, height))

    for i, (x, original_size_list, _) in enumerate(video_loader):
        start_time = time.time()
        if gpu:
            x = x.float().cuda()
        y = model(x)
        if not probability_output: y = nn.Sigmoid()(y)
        if gpu:
            y = y.detach().cpu()
        y = y.numpy()
        for j, prediction in enumerate(y[:, 0, :, :]):
            original_size = original_size_list[j].numpy()
            if not probability_output:
                img_data = post_process_png(prediction, original_size)
            else:
                img_data = post_process_probability2(prediction, original_size)
            img_data = cv2.resize(img_data, (width, height))
            # Using min-max normalization with cv2 constants
            img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX)
            img_data = img_data.astype(np.uint8)
            img_data = cv2.applyColorMap(img_data, cv2.COLORMAP_JET)
            out.write(img_data)

            # Overlay the saliency map on the original frame
            original_frame = frames[i * batch_size + j]
            overlay_frame = cv2.addWeighted(original_frame, 0.75, img_data, 0.25, 0)
            out_overlay.write(overlay_frame)

        end_time = time.time()
        print(f"Frame {i * batch_size + j + 1}/{frame_count} done, time taken: {end_time - start_time:.2f} seconds")

    out.release()
    out_overlay.release()

    end_total_time = time.time()  # End the timer
    total_time = end_total_time - start_total_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Frames per second: {frame_count / total_time:.2f}")
    
# ...existing code...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configs for predict.')
    parser.add_argument('-model_type', action='store', dest='model_type',
                        help='model type can be either Concatenation or Addition', default='A')
    parser.add_argument('-finetune_dataset', action='store', dest='finetune_dataset',
                        help='Dataset that the model fine tuned on.', default='SALICON')
    parser.add_argument('-use_webcam', action='store_true', dest='use_webcam',
                        help='Use webcam instead of video file')
    parser.add_argument('-input_path', action='store', dest='input_path',
                        help='path to input video (not needed for webcam)')
    parser.add_argument('-output_path', action='store', dest='output_path',
                        help='path to output video (not needed for webcam)')
    parser.add_argument('-batch_size', action='store', dest='batch_size',
                        help='batch size.', default=1, type=int)
    parser.add_argument('-probability_output', action='store', dest='probability_output',
                        help='use probability_output or not', default=False, type=bool)
    parser.add_argument('-gpu', action='store', dest='gpu',
                        help='use gpu or not', default=True, type=bool)
    parser.add_argument('-threshold', action='store', dest='threshold',
                        help='initial saliency threshold (0.0-1.0)', default=0.3, type=float)
    
    args = parser.parse_args()

    # Ensure the script is running from the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != script_dir:
        os.chdir(script_dir)
        print(f"Changed working directory to {script_dir}")

    # Run webcam with threshold
    print("Starting webcam prediction with masking (press 'q' to exit, +/- to adjust threshold)...")
    predict_webcam(
        model_type=args.model_type,
        finetune_dataset=args.finetune_dataset,
        probability_output=args.probability_output,
        batch_size=args.batch_size,
        gpu=args.gpu,
        initial_threshold=args.threshold
    )