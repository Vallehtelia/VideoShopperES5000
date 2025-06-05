#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI script to generate TikTok-ready (9:16) thumbnail images from every video in root/processed/.
Each thumbnail:
  1. Captures the first frame of the video.
  2. Resizes the frame’s width to 1080px (keeping aspect ratio).
  3. Centers it on a black 1080×1920 canvas (letterboxed top/bottom).
  4. Dynamically scales a two-line overlay so it fills most of the canvas width:
       VIDEONAME
       PART <n>
     using a TrueType font (supports Ä, Ö, etc.).
Thumbnails are saved as JPEGs to root/images/, named “{basename}.jpg”.

Video filenames must follow “NÖRTTI I-1.mp4”, “NÖRTTI II-3.mp4”, etc.
where “I”, “II”, “III”, “IV” pick the background color and “-<number>” is the part.
Run:
    python3 root/scripts/create_thumbnails.py
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
PROCESSED_DIR = os.path.join(ROOT_DIR, "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to a TrueType font that supports Finnish characters (Ä, Ö, etc.)
# Adjust this path if needed (e.g., to a bundled TTF in your project).
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Lighter BGR colors for each Roman numeral part
COLOR_MAP = {
    "I":   (180, 100, 255),  # lighter red/pink
    "II":  (255, 180, 100),  # lighter cyan
    "III": (200,  50, 200),  # lighter purple
    "IV":  (100, 255, 100),  # lighter green
}
DEFAULT_COLOR = (50,  50,  50)  # dark gray if parsing fails

# Target TikTok thumbnail dimensions
CANVAS_W = 1080
CANVAS_H = 1920


def parse_filename(fname):
    """
    Given a filename like 'NÖRTTI I-1.mp4' or 'NÖRTTI III-12.mp4',
    split on the last hyphen to extract <VIDEONAME> (with Roman) and part number.
    Returns (video_name, roman, part_str).
    """
    name, _ext = os.path.splitext(fname)
    if '-' in name:
        left, part_str = name.rsplit('-', 1)
        part_str = part_str.strip()
        if ' ' in left:
            base, roman = left.rsplit(' ', 1)
            if roman in COLOR_MAP:
                video_name = f"{base} {roman}"
                return video_name, roman, part_str
    # Fallback: no recognized pattern
    return name, None, None


def letterbox_frame(frame):
    """
    1. Resize frame so its width = CANVAS_W (1080px), keeping aspect ratio.
    2. Create a black CANVAS_H×CANVAS_W background.
    3. Vertically center the resized frame on that canvas.
    Returns the final 1080×1920 image, y_offset of frame start, and frame height.
    """
    h0, w0 = frame.shape[:2]
    scale = CANVAS_W / w0
    new_h = int(h0 * scale)
    resized = cv2.resize(frame, (CANVAS_W, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Compute vertical offset to center the resized frame
    y_offset = (CANVAS_H - new_h) // 2
    canvas[y_offset : y_offset + new_h, 0 : CANVAS_W] = resized
    return canvas, y_offset, new_h


def get_font_size(text_lines, target_w, font_path):
    """
    Estimate a font size so that the widest text line is approximately target_w pixels.
    """
    test_size = 90
    font = ImageFont.truetype(font_path, test_size)
    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    base_widths = []
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        base_widths.append(w)
    max_base = max(base_widths)
    scale_factor = (target_w / max_base)
    return max(int(test_size * scale_factor), 1)


def put_unicode_text(
    cv_img: np.ndarray,
    text_lines: list[str],
    font_path: str,
    font_size: int,
    box_color_bgr: tuple[int,int,int],
    box_coords: tuple[int,int,int,int]
) -> np.ndarray:
    """
    Draws an opaque rounded rectangle background and renders Unicode text (multiple lines) onto cv_img.

    - cv_img:       BGR numpy array (OpenCV image)
    - text_lines:   list of strings to draw (e.g. ["NÖRTTI I", "PART 1"])
    - font_path:    path to a .ttf font file supporting Unicode
    - font_size:    point size for the font
    - box_color_bgr: BGR tuple for rectangle fill
    - box_coords:   (x, y, width, height) of the rectangle
    Returns the modified BGR numpy array.
    """
    # 1) Convert cv_img → PIL so we can use PIL's Unicode font drawing:
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    x, y, w_box, h_box = box_coords

    # 2) Compute a good contrasting text color (white or black) from the BGR box color:
    #    Convert BGR → RGB first:
    r_box, g_box, b_box = box_color_bgr[2], box_color_bgr[1], box_color_bgr[0]
    #    Compute a simple “luma”:
    luma = 0.299*r_box + 0.587*g_box + 0.114*b_box
    if luma < 128:
        text_fill = (255, 255, 255)   # white text on dark box
    else:
        text_fill = (  0,   0,   0)   # black text on light box

    # 3) Draw the rounded rectangle itself (fill=box_color in RGB order):
    radius = 20
    draw.rounded_rectangle(
        [x, y, x + w_box, y + h_box],
        radius=radius,
        fill=(r_box, g_box, b_box)
    )

    # 4) Re‐measure each line’s width & height so that we can center them inside the box:
    widths: list[int] = []
    heights: list[int] = []
    spacing = int(font_size * 0.1)  # 10% of font size between lines
    total_h = 0
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        widths.append(w)
        heights.append(h)
        total_h += h
    total_h += spacing * (len(text_lines) - 1)

    # 5) Vertically center the block of text inside the rectangle:
    current_y = y + (h_box - total_h) // 2
    for i, line in enumerate(text_lines):
        text_w = widths[i]
        text_h = heights[i]
        text_x = x + (w_box - text_w) // 2
        draw.text((text_x, current_y), line, font=font, fill=text_fill)
        current_y += text_h + spacing

    # 6) Convert back to OpenCV‐style BGR and return:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def create_thumbnail(
    video_path: str,
    color_bgr: tuple[int,int,int],
    text_lines: list[str],
    output_path: str,
):
    """
    1. Capture the first frame of video_path.
    2. Letterbox it into a 1080×1920 black canvas.
    3. Dynamically compute a font size so text fills ~90% of canvas width.
    4. Draw an opaque rounded rectangle behind the text (inside top letterbox) 
       and render two lines in a high‐contrast color.
    5. Save as JPEG to output_path.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open {video_path}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Warning: Could not read frame from {video_path}")
        return

    # STEP 1 & 2: Resize width→1080 and letterbox into 1080×1920
    canvas, top_bar_height, frame_height = letterbox_frame(frame)
    h_canvas, w_canvas = canvas.shape[:2]  # should be (1920, 1080)

    # STEP 3: Dynamically find a font_size so that text width ≈ 0.9 * CANVAS_W:
    target_text_w = int(0.9 * w_canvas)
    font_size = get_font_size(text_lines, target_text_w, FONT_PATH)

    # STEP 4a: Recompute each line’s exact pixel‐size (with PIL) to figure out rectangle size:
    pil_font = ImageFont.truetype(FONT_PATH, font_size)
    dummy_img = Image.new("RGB", (10, 10))
    draw_dummy = ImageDraw.Draw(dummy_img)
    widths: list[int] = []
    heights: list[int] = []
    line_spacing = int(font_size * 0.1)
    for line in text_lines:
        bbox = draw_dummy.textbbox((0, 0), line, font=pil_font)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]
        widths.append(w_text)
        heights.append(h_text)
    text_block_w = max(widths)
    text_block_h = sum(heights) + (len(text_lines) - 1) * line_spacing

    #  STEP 4b: Add padding around the text so it never “touches” the rectangle border
    padding_x = 30
    padding_y = 30
    box_w = text_block_w + 2 * padding_x
    box_h = text_block_h + 2 * padding_y

    #  STEP 4c: Place that pill “20px from the top” and horizontally centered
    box_x = (w_canvas - box_w) // 2
    box_y = 20

    # STEP 5: Draw rounded rectangle + Unicode text via our updated function
    canvas = put_unicode_text(
        canvas,
        text_lines,
        FONT_PATH,
        font_size,
        color_bgr,
        (box_x, box_y, box_w, box_h)
    )

    # STEP 6: Save as JPEG
    cv2.imwrite(output_path, canvas)
    print(f"Saved thumbnail: {output_path}")


def main():
    files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.lower().endswith(".mp4"))
    if not files:
        print(f"No .mp4 files found in {PROCESSED_DIR}")
        sys.exit(0)

    for fname in files:
        video_name, roman, part_str = parse_filename(fname)
        if roman and part_str:
            color = COLOR_MAP.get(roman, DEFAULT_COLOR)
            text_lines = [video_name, f"PART {part_str}"]
        else:
            color = DEFAULT_COLOR
            name_only = os.path.splitext(fname)[0]
            text_lines = [name_only]

        video_path = os.path.join(PROCESSED_DIR, fname)
        output_fname = f"{os.path.splitext(fname)[0]}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_fname)

        create_thumbnail(video_path, color, text_lines, output_path)

if __name__ == "__main__":
    main()