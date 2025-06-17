#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
import shutil
import subprocess
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Paths
def get_paths():
    root = Path(__file__).resolve().parent.parent
    return {
        "processed": root / "processed",
        "cartoonized": root / "cartoonized",
        "frames": root / "frames"
    }

paths = get_paths()
for p in paths.values():
    p.mkdir(parents=True, exist_ok=True)

# Ensure GPU available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please run this script on a machine with a GPU.")

# Load model with 8-bit quantization to reduce VRAM usage
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "AnalogMutations/cartoonizer",
    torch_dtype=torch.float16,
    load_in_8bit=True,          # quantize weights to 8-bit
    device_map="balanced"         # automatically place layers on GPU
)
pipe.enable_attention_slicing()
# Enable memory-efficient attention and CPU offload to reduce peak GPU VRAM usage
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass  # xformers might not be installed

# Offload model layers to CPU sequentially
pipe.reset_device_map()
pipe.enable_model_cpu_offload()
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# Helper to resize
from PIL import Image as PILImage

def resize_image(img: PILImage, size=(512, 512)) -> PILImage:
    return img.resize(size, PILImage.BICUBIC)

# Process videos in processed directory
target_files = sorted(paths["processed"].glob("*.mp4"))
if not target_files:
    raise FileNotFoundError("No .mp4 files found in processed directory.")

BATCH_SIZE = 4
prompts = ["Cartoonize" for _ in range(BATCH_SIZE)]

for video_path in target_files:
    basename = video_path.stem
    print(f"\nProcessing: {basename}")

    # Extract frames
    dir_frames = paths["frames"] / basename
    shutil.rmtree(dir_frames, ignore_errors=True)
    dir_frames.mkdir(parents=True)
    pattern = str(dir_frames / "frame_%05d.png")
    subprocess.run(["ffmpeg", "-i", str(video_path), pattern], check=True)

    # Cartoonify in batches
    frame_paths = sorted(dir_frames.glob("frame_*.png"))
    total = len(frame_paths)
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = frame_paths[i:i + BATCH_SIZE]
        imgs = [resize_image(Image.open(fp).convert("RGB")) for fp in batch]
        outs = pipe(prompt=prompts[:len(batch)], image=imgs).images
        for out_img, fp in zip(outs, batch):
            out_img.save(fp)

        elapsed = time.time() - start_time
        done = i + len(batch)
        fps = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / fps if fps > 0 else 0
        tqdm.write(f"ETA: {eta:.1f}s ({done}/{total})")

    # Reassemble video
    print("Combining frames to video...")
    temp_vid = paths["cartoonized"] / f"{basename}_temp.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15", "-i", str(dir_frames / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(temp_vid)
    ], check=True)

    # Merge audio
    final_vid = paths["cartoonized"] / f"{basename}_cartoonized.mp4"
    print("Merging audio...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(temp_vid), "-i", str(video_path),
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(final_vid)
    ], check=True)

    print(f"Done: {final_vid}")
