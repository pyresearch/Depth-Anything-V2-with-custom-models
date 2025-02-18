import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
import supervision as sv  # Import supervision library
import pyresearch

def process_video(video_path, encoder, input_size, outdir, pred_only, grayscale):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Depth Anything Model Setup
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the Depth Anything model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Initialize YOLO model (replace 'best.pt' with the actual model weight path)
    yolo_model = YOLO('newbest.pt')  # Use YOLOv8 model
    yolo_model.to(DEVICE)

    # Initialize supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()  # Updated to BoxAnnotator
    label_annotator = sv.LabelAnnotator()

    # If video_path is a file, make a single-element list; otherwise search recursively
    filenames = [video_path] if os.path.isfile(video_path) else glob.glob(os.path.join(video_path, '**/*'), recursive=True)
    os.makedirs(outdir, exist_ok=True)  # Ensure the output directory exists
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
        # Define output video paths
        base_name = os.path.splitext(os.path.basename(filename))[0]
        detection_output_path = os.path.join(outdir, f'{base_name}_detection.mp4')
        depth_output_path = os.path.join(outdir, f'{base_name}_depth.mp4')
        combined_output_path = os.path.join(outdir, f'{base_name}_combined.mp4')

        # Initialize VideoWriters for detection, depth, and combined videos
        detection_writer = cv2.VideoWriter(detection_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        depth_writer = cv2.VideoWriter(depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        combined_writer = cv2.VideoWriter(combined_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width * 2, frame_height))
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            # Resize frame to input_size for processing
            resized_frame = cv2.resize(raw_frame, (input_size, input_size))

            # Apply YOLO detection
            results = yolo_model(resized_frame)  # YOLOv8 detection (results contains bounding boxes)

            # Convert YOLO results into supervision detection format
            detections = sv.Detections.from_ultralytics(results[0])

            # Annotate with bounding boxes and labels
            annotated_frame = bounding_box_annotator.annotate(scene=resized_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            # Resize annotated_frame back to original dimensions
            annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

            # Apply depth inference
            depth = depth_anything.infer_image(resized_frame, input_size)
            depth = np.uint8((depth - depth.min()) / (depth.max() - depth.min()) * 255)

            if grayscale:
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)  # Convert to grayscale if requested
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)  # Apply color map for depth

            # Resize depth to match original frame dimensions
            depth = cv2.resize(depth, (frame_width, frame_height))

            # Combine YOLO detection and depth map
            combined_frame = np.hstack((annotated_frame, depth))

            # Write frames to respective videos
            detection_writer.write(annotated_frame)  # Save detection video
            depth_writer.write(depth)  # Save depth video
            combined_writer.write(combined_frame)  # Save combined video

            # Display frames (optional)
            cv2.imshow('YOLO Detection', annotated_frame)
            cv2.imshow('Depth Map', depth)
            cv2.imshow('Combined Frame', combined_frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        raw_video.release()
        detection_writer.release()
        depth_writer.release()
        combined_writer.release()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 with YOLOv8')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video file or directory containing videos.')
    parser.add_argument('--input-size', type=int, default=640, help='Input size for depth inference.')
    parser.add_argument('--outdir', type=str, default='./vis_video_depth', help='Output directory where the processed video will be saved.')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Encoder type for Depth Anything V2.')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only display/save the prediction (depth map).')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='Display the depth map in grayscale.')
    args = parser.parse_args()

    process_video(args.video_path, args.encoder, args.input_size, args.outdir, args.pred_only, args.grayscale)