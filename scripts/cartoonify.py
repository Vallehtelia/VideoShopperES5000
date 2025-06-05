#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import shutil
import subprocess
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "processed"
CARTOONIZED_DIR = ROOT_DIR / "cartoonized"
FRAMES_DIR = ROOT_DIR / "frames"

CARTOONIZED_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please run this script on a machine with a GPU.")

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "AnalogMutations/cartoonizer", torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

video_files = sorted(PROCESSED_DIR.glob("*.mp4"))[:3]
if not video_files:
    raise FileNotFoundError("No .mp4 files found in ~/processed/")

for video_path in video_files:
    basename = video_path.stem
    output_video = CARTOONIZED_DIR / f"{basename}_cartoonized.mp4"

    print(f"\nProcessing: {basename}")
    print("Extracting frames...")
    frames_out_dir = FRAMES_DIR / basename
    shutil.rmtree(frames_out_dir, ignore_errors=True)
    frames_out_dir.mkdir(parents=True)
    frame_pattern = str(frames_out_dir / "frame_%05d.png")
    subprocess.run(["ffmpeg", "-i", str(video_path), frame_pattern], check=True)

    frame_files = sorted(frames_out_dir.glob("frame_*.png"))
    total_frames = len(frame_files)
    start_time = time.time()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.ToPILImage()
    ])

    print("Cartoonifying frames...")
    for idx, frame_path in enumerate(tqdm(frame_files)):
        img = Image.open(frame_path).convert("RGB")
        img = transform(img)
        image_out = pipe(prompt="Cartoonize the following image", image=img).images[0]
        image_out.save(frame_path)

        elapsed = time.time() - start_time
        fps = (idx + 1) / elapsed
        eta = (total_frames - idx - 1) / fps if fps > 0 else 0
        tqdm.write(f"ETA: {eta:.2f} seconds")

    print("Combining frames back to video...")
    cartoon_temp = CARTOONIZED_DIR / f"{basename}_temp.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "25", "-i", str(frames_out_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cartoon_temp)
    ], check=True)

    print("Merging audio from original video...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(cartoon_temp), "-i", str(video_path),
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output_video)
    ], check=True)

    print(f"Done. Cartoonized video saved to: {output_video}")