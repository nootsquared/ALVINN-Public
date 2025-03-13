import model.fastSal as fastsal  # Will need to ensure this module exists
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
from utils import load_weight

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

def post_process_png(prediction, original_size):
    """Post-process the model's prediction to create a saliency map."""
    # Resize the prediction to the original image size
    resized_prediction = cv2.resize(prediction, (original_size[0], original_size[1]))
    # Normalize to 0-1 range
    normalized_prediction = (resized_prediction - resized_prediction.min()) / (resized_prediction.max() - resized_prediction.min() + 1e-8)
    # Scale to 0-255 for visualization
    return (normalized_prediction * 255).astype(np.uint8)

def post_process_probability2(prediction, original_size):
    """Alternative post-processing for probability output."""
    # Resize the prediction to the original image size
    resized_prediction = cv2.resize(prediction, (original_size[0], original_size[1]))
    # Ensure values are in valid range (0-1)
    clipped_prediction = np.clip(resized_prediction, 0, 1)
    # Scale to 0-255 for visualization
    return (clipped_prediction * 255).astype(np.uint8)


def predict_webcam(model_type, finetune_dataset, probability_output, batch_size, gpu=True):
    start_total_time = time.time()  # Start the timer

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
                    img_data = post_process_png(prediction, original_size)
                else:
                    img_data = post_process_probability2(prediction, original_size)
                img_data = cv2.resize(img_data, (width, height))
                # Using min-max normalization with cv2 constants
                img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX)
                img_data = img_data.astype(np.uint8)
                img_data = cv2.applyColorMap(img_data, cv2.COLORMAP_JET)

                # Overlay the saliency map on the original frame
                overlay_frame = cv2.addWeighted(frame, 0.75, img_data, 0.25, 0)

                # Calculate and display FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(overlay_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display the frame
                cv2.imshow('Webcam Live Stream', overlay_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

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
    args = parser.parse_args()

    # Ensure the script is running from the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != script_dir:
        os.chdir(script_dir)
        print(f"Changed working directory to {script_dir}")

    # Always run webcam by default (ignore other arguments)
    print("Starting webcam prediction (press 'q' to exit)...")
    predict_webcam(
        model_type=args.model_type,
        finetune_dataset=args.finetune_dataset,
        probability_output=args.probability_output,
        batch_size=args.batch_size,
        gpu=args.gpu
    )