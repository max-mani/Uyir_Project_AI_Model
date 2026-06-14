"""Shared incident clip extraction and encoding utilities."""

import math
import os
import shutil
import subprocess
import sys

import cv2

import config


def transcode_video_for_browser(video_path):
    """Re-encode OpenCV output to H.264 so browsers can play it in <video>."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[WARN] ffmpeg not found; browser may not play processed video.")
        return False

    temp_path = f"{video_path}.browser.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        temp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if (
        result.returncode == 0
        and os.path.exists(temp_path)
        and os.path.getsize(temp_path) > 0
    ):
        os.replace(temp_path, video_path)
        return True

    print("[WARN] ffmpeg transcode failed:", result.stderr.strip())
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return False


def create_video_writer(output_path, fps, width, height):
    """Create a VideoWriter using a codec/backend available on the current OS."""
    if sys.platform == "win32":
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, cv2.CAP_MSMF, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer

    raise RuntimeError(
        "Failed to initialize video writer. "
        "Install ffmpeg/opencv with video encoding support and retry."
    )


def ensure_clip_browser_ready(path):
    """Transcode clip to browser-friendly H.264 if possible."""
    return transcode_video_for_browser(path)


def extract_clip_from_file(
    source_path,
    center_frame,
    fps,
    output_path,
    before_sec=None,
    after_sec=None,
    total_frames=None,
):
    """
    Extract a clip centered on center_frame using ffmpeg (source video time base).
    Returns True when a clip file was written.
    """
    before_sec = before_sec if before_sec is not None else config.CLIP_SECONDS_BEFORE
    after_sec = after_sec if after_sec is not None else config.CLIP_SECONDS_AFTER

    if fps <= 0 or math.isnan(fps):
        fps = 30.0

    start_frame = max(0, int(center_frame - before_sec * fps))
    if total_frames is not None and total_frames > 0:
        end_frame = min(int(total_frames - 1), int(center_frame + after_sec * fps))
    else:
        end_frame = int(center_frame + after_sec * fps)

    if end_frame <= start_frame:
        end_frame = start_frame + 1

    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame + 1) / fps

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            source_path,
            "-t",
            f"{duration_sec:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if (
            result.returncode == 0
            and os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        ):
            return True
        print("[WARN] ffmpeg clip extract failed:", result.stderr.strip())

    return _extract_clip_opencv(source_path, start_frame, end_frame, fps, output_path)


def _extract_clip_opencv(source_path, start_frame, end_frame, fps, output_path):
    """Fallback clip extraction when ffmpeg is unavailable."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = create_video_writer(output_path, fps, width, height)
    frames_written = 0

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frames_written += 1

    cap.release()
    writer.release()

    if frames_written == 0 or not os.path.exists(output_path):
        return False

    ensure_clip_browser_ready(output_path)
    return True


def write_clip_from_frames(frames, fps, output_path):
    """Assemble a list of BGR frames into an MP4 clip."""
    if not frames:
        return False

    if fps <= 0:
        fps = float(config.CLIP_BUFFER_FPS)

    height, width = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = create_video_writer(output_path, fps, width, height)

    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False

    ensure_clip_browser_ready(output_path)
    return True
