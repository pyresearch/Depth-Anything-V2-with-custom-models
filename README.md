# Depth-Anything-V2-with-custom-models
Depth Anything V2


# Running script on images

python run.py --encoder vitl --img-path assets/examples --outdir depth_vis


# Running script on videos

python run_video.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis --input-size 256 --pred-only --grayscale
