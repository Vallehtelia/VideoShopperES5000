#!/usr/bin/env python3
"""
CLI script to download YouTube videos from a list in JSON and save them to root/videos/.
Place this script in root/scripts/fetch_video.py and run it from the command line.
"""
import os
import sys
import shutil
import json

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("yt-dlp is not installed. Please install it with 'pip install yt-dlp'.")
    sys.exit(1)


def ffmpeg_installed():
    return shutil.which('ffmpeg') is not None


def load_urls(json_path):
    if not os.path.exists(json_path):
        print(f"URL list file not found: {json_path}")
        sys.exit(1)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("URL JSON must contain a list of URLs.")
            sys.exit(1)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        sys.exit(1)


def download_video(url, output_dir, use_merge):
    # yt-dlp options
    if use_merge:
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'format': 'bestvideo+bestaudio/best',
        }
    else:
        print("Warning: ffmpeg not found. Downloading best single-format video for URL: {url}")
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'format': 'best',
        }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading: {url}")
            ydl.download([url])
        print(f"Completed: {url}\n")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def main():
    # Path to URL list JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    url_json_path = os.path.join(root_dir, "URL", "url.json")

    # Load URLs
    urls = load_urls(url_json_path)
    if not urls:
        print("No URLs found in JSON file.")
        sys.exit(0)

    # Output directory: root/videos/
    output_dir = os.path.join(root_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)

    # Check ffmpeg availability once
    use_merge = ffmpeg_installed()

    for url in urls:
        download_video(url, output_dir, use_merge)

    print("All downloads processed.")

if __name__ == "__main__":
    main()