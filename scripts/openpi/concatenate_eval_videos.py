#!/usr/bin/env python3
"""Concatenate all OpenPI evaluation videos into a single video.

Reads all episode_*.mp4 files from the eval video directory, sorts them
chronologically, and writes a single concatenated video. Each episode is
annotated with its episode number, success/failure label, and instruction
text from the companion JSON metadata.

Usage:
    python concatenate_eval_videos.py [--video-dir DIR] [--output PATH] [--fps FPS]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_metadata(video_path: Path) -> dict | None:
    meta_path = video_path.with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def draw_label(frame: np.ndarray, text: str, success: bool | None) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    bar_height = 32
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if success is True:
        color = (0, 220, 0)
    elif success is False:
        color = (0, 0, 220)
    else:
        color = (255, 255, 255)

    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Concatenate OpenPI eval videos")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="~/jellyho/openpi_eval_videos",
        help="Directory containing episode_*.mp4 files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: <video-dir>/concatenated.mp4)",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--speed", type=float, default=1.0, help="Speedup factor (e.g. 4 for 4x)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir).expanduser()
    if not video_dir.exists():
        print(f"Error: directory not found: {video_dir}")
        return

    video_files = sorted(video_dir.glob("episode_*.mp4"))
    if not video_files:
        print(f"No episode_*.mp4 files found in {video_dir}")
        return

    output_path = Path(args.output) if args.output else video_dir / "concatenated.mp4"
    print(f"Found {len(video_files)} episode videos in {video_dir}")

    ref_cap = cv2.VideoCapture(str(video_files[0]))
    out_w = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Error: could not open video writer for {output_path}")
        return

    total_frames = 0
    for vf in video_files:
        meta = load_metadata(vf)
        if meta:
            label = "SUCCESS" if meta.get("success") else "FAILURE"
            ep = meta.get("episode", "?")
            instruction = meta.get("instruction", "")
            text = f"Ep {ep} [{label}] {instruction}"
            success = meta.get("success")
        else:
            text = vf.stem
            success = None

        skip = max(1, int(args.speed))
        cap = cv2.VideoCapture(str(vf))
        frame_idx = 0
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                frame = draw_label(frame, text, success)
                writer.write(frame)
                frame_count += 1
            frame_idx += 1
        cap.release()
        total_frames += frame_count
        print(f"  {vf.name}: {frame_count} frames ({label if meta else 'no metadata'})")

    writer.release()
    duration = total_frames / args.fps if args.fps > 0 else 0
    print(f"\nSaved: {output_path}")
    print(f"Total: {total_frames} frames, {duration:.1f}s at {args.fps} fps")


if __name__ == "__main__":
    main()
