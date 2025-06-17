#!/usr/bin/env python3
"""
CLI script to cut videos from root/videos/ into overlapping 30-second segments and save to root/processed/.
Segments overlap by 1 second. If the final remaining part is shorter than 20 seconds, it is merged into the previous segment.
This version re-encodes video at 15 fps to reduce frame rate.
Place this script in root/scripts/cut_videos.py and run it from the command line.
"""
import os
import sys
import shutil
import subprocess

SEGMENT_LENGTH = 40  # seconds
OVERLAP = 1          # seconds
STEP = SEGMENT_LENGTH - OVERLAP
MIN_REMAINDER = 20   # seconds
TARGET_FPS = 15      # frames per second


def ffmpeg_installed():
    return shutil.which('ffmpeg') is not None


def ffprobe_installed():
    return shutil.which('ffprobe') is not None


def get_duration(path):
    try:
        result = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], stderr=subprocess.STDOUT)
        return float(result.strip())
    except Exception as e:
        print(f"Error retrieving duration for {path}: {e}")
        sys.exit(1)


def make_segments(duration):
    starts = []
    t = 0
    while t < duration:
        starts.append(t)
        t += STEP
    segments = []
    for s in starts:
        end = min(s + SEGMENT_LENGTH, duration)
        segments.append((s, end))
    # Merge last if too short
    if len(segments) > 1:
        last_start, last_end = segments[-1]
        if (last_end - last_start) < MIN_REMAINDER:
            prev_start, _ = segments[-2]
            segments[-2] = (prev_start, duration)
            segments.pop()
    return segments


def cut_video(input_path, output_dir):
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    duration = get_duration(input_path)
    segments = make_segments(duration)

    for idx, (start, end) in enumerate(segments, start=1):
        seg_duration = end - start
        output_name = f"{name}-{idx}.mp4"
        output_path = os.path.join(output_dir, output_name)
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(seg_duration),
            '-vf', f'fps={TARGET_FPS}',  # re-encode at target fps
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',
            output_path
        ]
        print(f"Creating segment {idx}: {start:.2f}s to {end:.2f}s at {TARGET_FPS}fps -> {output_name}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating segment {idx} for {filename}: {e}")


def main():
    if not ffmpeg_installed() or not ffprobe_installed():
        print("ffmpeg and/or ffprobe not found. Please install FFmpeg.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    videos_dir = os.path.join(root_dir, 'videos')
    processed_dir = os.path.join(root_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.isdir(videos_dir):
        print(f"Videos directory not found: {videos_dir}")
        sys.exit(1)

    for fname in os.listdir(videos_dir):
        input_path = os.path.join(videos_dir, fname)
        if os.path.isfile(input_path):
            print(f"Processing video: {fname}")
            cut_video(input_path, processed_dir)

    print("All videos processed.")


if __name__ == "__main__":
    main()
