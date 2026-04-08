import os
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
from mtcnn import MTCNN
import tensorflow as tf

# Suppress TensorFlow logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize MTCNN detector
detector = MTCNN()

def get_filename_only(file_path):
    return Path(file_path).stem

def extract_frames(video_path, output_dir, sample_rate=1):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames.append(frame_filename)
        frame_count += 1
    cap.release()
    return saved_frames

def crop_faces_from_frames(frame_paths, faces_dir, margin_ratio=0.3, min_confidence=0.95):
    os.makedirs(faces_dir, exist_ok=True)
    face_paths = []
    for frame_path in frame_paths:
        image_bgr = cv2.imread(frame_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image_rgb)

        for i, result in enumerate(results):
            conf = result['confidence']
            if len(results) >= 2 and conf < min_confidence:
                print(f"Skipped low confidence face ({conf:.2f}) in {frame_path}")
                continue

            x, y, w, h = result['box']
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image_rgb.shape[1], x + w + margin_x)
            y2 = min(image_rgb.shape[0], y + h + margin_y)

            crop = image_rgb[y1:y2, x1:x2]
            base = get_filename_only(frame_path)
            face_filename = os.path.join(faces_dir, f"{base}_face{i:02d}.png")
            cv2.imwrite(face_filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            face_paths.append(face_filename)

    return face_paths

def process_video(video_path, output_zip=None, sample_rate=1):
    """
    Main function: extract faces from video and create a ZIP file.
    If output_zip is None, creates a zip in the same directory as video.
    Returns the path to the created ZIP file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = os.path.join(tmpdir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        print(f"Extracting frames from {video_path}...")
        frame_paths = extract_frames(video_path, frames_dir, sample_rate)
        if not frame_paths:
            raise RuntimeError("No frames could be extracted from the video")

        faces_dir = os.path.join(tmpdir, 'faces')
        os.makedirs(faces_dir, exist_ok=True)

        print("Detecting and cropping faces...")
        face_paths = crop_faces_from_frames(frame_paths, faces_dir)
        if not face_paths:
            raise RuntimeError("No faces detected in the video")

        # Determine output zip path
        if output_zip is None:
            video_name = get_filename_only(video_path)
            output_zip = os.path.join(os.path.dirname(video_path), f"{video_name}_faces.zip")

        # Create ZIP archive
        with zipfile.ZipFile(output_zip, 'w') as zf:
            for face in face_paths:
                zf.write(face, arcname=os.path.basename(face))

        print(f"✅ Faces saved to: {output_zip}")
        return output_zip

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract faces from a video using MTCNN")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output ZIP file path (default: video_folder/video_name_faces.zip)")
    parser.add_argument("-r", "--sample-rate", type=int, default=1, help="Process every Nth frame (default: 1 = all frames)")
    args = parser.parse_args()

    try:
        process_video(args.video, args.output, args.sample_rate)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)