#!/usr/bin/env python3
"""
CLI script to enhance video quality using AI-based super-resolution (EDSR x4) with GPU (CUDA) if available, otherwise falls back to CPU.
Place the pre-trained model "EDSR_x4.pb" in root/models/.
The script reads videos from root/processed/, applies super-resolution, estimates processing time per video, and saves enhanced videos to root/enhanced/.
Run: python3 root/scripts/enhance_videos.py
"""
import os
import sys
import shutil
import subprocess
import time
import cv2

# Dependencies:
#   - Python packages:
#       opencv-contrib-python (for CPU-only) or custom build opencv-contrib-python with CUDA support
#       numpy
#   - System packages:
#       ffmpeg
#       CUDA toolkit & drivers (if using GPU acceleration)


def ffmpeg_installed():
    return shutil.which('ffmpeg') is not None


def check_model(model_path):
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)


def estimate_time(frame_count, mode='cpu'):
    # Rough estimate: CPU ~ 0.1s per frame, GPU ~ 0.01s per frame (varies by hardware)
    per_frame = 0.01 if mode == 'gpu' else 0.1
    total_seconds = frame_count * per_frame
    return total_seconds


def enhance_video_cpu(input_path, output_path, sr):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Output dimensions: x4
    out_width = width * 4
    out_height = height * 4

    estimated = estimate_time(frame_count, mode='cpu')
    print(f"[CPU] Processing: {os.path.basename(input_path)} | Frames: {frame_count} | Estimated time: {estimated:.1f}s")

    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    start_time = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = sr.upsample(frame)
        out.write(result)
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {count}/{frame_count} frames ({elapsed:.1f}s elapsed)")
    cap.release()
    out.release()
    total_elapsed = time.time() - start_time
    print(f"[CPU] Completed in {total_elapsed:.1f}s | Saved: {output_path}")


def enhance_video_gpu(input_path, output_path, sr):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Output dimensions: x4
    out_width = width * 4
    out_height = height * 4

    estimated = estimate_time(frame_count, mode='gpu')
    print(f"[GPU] Processing: {os.path.basename(input_path)} | Frames: {frame_count} | Estimated time: {estimated:.1f}s")

    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    start_time = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        result_gpu = sr.upsample(gpu_frame)
        result = result_gpu.download()
        out.write(result)
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {count}/{frame_count} frames ({elapsed:.1f}s elapsed)")
    cap.release()
    out.release()
    total_elapsed = time.time() - start_time
    print(f"[GPU] Completed in {total_elapsed:.1f}s | Saved: {output_path}")


def main():
    if not ffmpeg_installed():
        print("ffmpeg not found. Please install FFmpeg.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    processed_dir = os.path.join(root_dir, 'processed')
    enhanced_dir = os.path.join(root_dir, 'enhanced')
    models_dir = os.path.join(root_dir, 'models')

    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'EDSR_x4.pb')
    check_model(model_path)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel('edsr', 4)

    use_gpu = False
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            use_gpu = True
            print("CUDA-enabled OpenCV found. Using GPU acceleration.")
        except Exception:
            use_gpu = False
    if not use_gpu:
        print("CUDA-enabled OpenCV not found or initialization failed. Falling back to CPU.")
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    if not os.path.isdir(processed_dir):
        print(f"Processed directory not found: {processed_dir}")
        sys.exit(1)

    for fname in os.listdir(processed_dir):
        input_path = os.path.join(processed_dir, fname)
        if os.path.isfile(input_path) and fname.lower().endswith('.mp4'):
            name, ext = os.path.splitext(fname)
            output_name = f"{name}_enhanced{ext}"
            output_path = os.path.join(enhanced_dir, output_name)
            if use_gpu:
                enhance_video_gpu(input_path, output_path, sr)
            else:
                enhance_video_cpu(input_path, output_path, sr)

    print("All videos enhanced.")

if __name__ == '__main__':
    main()
