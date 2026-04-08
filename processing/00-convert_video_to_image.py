import sys
import os
import cv2
import math

if len(sys.argv) < 3:
    print("Usage: python 00-convert_video_to_image.py <video_path> <output_frames_dir>")
    sys.exit(1)

video_path = sys.argv[1]
output_frames_dir = sys.argv[2]

def get_filename_only(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def process_video(video_path, frames_base_path):
    """Convert a single video to frames."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False

    # Create frames subfolder named after the video
    video_name = get_filename_only(video_path)
    output_dir = os.path.join(frames_base_path, video_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % max(1, math.floor(frame_rate)) == 0:
            # Resize logic (same as yours)
            h, w = frame.shape[:2]
            if w < 300:
                scale = 2
            elif w > 1900:
                scale = 0.33
            elif 1000 < w <= 1900:
                scale = 0.5
            else:
                scale = 1

            new_w = int(w * scale)
            new_h = int(h * scale)
            new_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_filename = f"{video_name}-{count:03d}.png"
            cv2.imwrite(os.path.join(output_dir, out_filename), new_frame)
            count += 1

    cap.release()
    print(f"Done: {video_path} -> {count} frames saved to {output_dir}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 00-convert_video_to_image.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    # You can change this path or make it configurable
    frames_base_path = r"E:\Setu\flask\frames"

    success = process_video(video_path, frames_base_path)
    sys.exit(0 if success else 1)