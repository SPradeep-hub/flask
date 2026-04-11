import os
import sys
import tempfile
import zipfile
from pathlib import Path
import cv2
import numpy as np

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

def get_face_detector(method='dnn'):
    """Return a face detector function."""
    if method == 'dnn':
        # Download model files if not present (only once)
        proto_path = 'deploy.prototxt'
        model_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            print("Downloading DNN face detector models...")
            import urllib.request
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/deploy.prototxt',
                proto_path
            )
            urllib.request.urlretrieve(
                'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel',
                model_path
            )
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        def detect_faces(image):
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    faces.append({'box': (x, y, x2-x, y2-y), 'confidence': confidence})
            return faces
        return detect_faces
    else:  # Haar cascade (simpler, less accurate)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        def detect_faces(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return [{'box': (x, y, w, h), 'confidence': 1.0} for (x, y, w, h) in rects]
        return detect_faces

def crop_faces_from_frames(frame_paths, faces_dir, margin_ratio=0.3, detector=None):
    os.makedirs(faces_dir, exist_ok=True)
    if detector is None:
        detector = get_face_detector('dnn')
    face_paths = []
    for frame_path in frame_paths:
        image = cv2.imread(frame_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector(rgb)
        for i, res in enumerate(results):
            x, y, w, h = res['box']
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(rgb.shape[1], x + w + margin_x)
            y2 = min(rgb.shape[0], y + h + margin_y)
            crop = rgb[y1:y2, x1:x2]
            base = get_filename_only(frame_path)
            out_name = f"{base}_face{i:02d}.png"
            out_path = os.path.join(faces_dir, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            face_paths.append(out_path)
    return face_paths

def process_video(video_path, output_zip=None, sample_rate=1, detector_method='dnn'):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    detector = get_face_detector(detector_method)
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = os.path.join(tmpdir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Extracting frames from {video_path}...")
        frame_paths = extract_frames(video_path, frames_dir, sample_rate)
        if not frame_paths:
            raise RuntimeError("No frames could be extracted")
        faces_dir = os.path.join(tmpdir, 'faces')
        os.makedirs(faces_dir, exist_ok=True)
        print("Detecting and cropping faces...")
        face_paths = crop_faces_from_frames(frame_paths, faces_dir, detector=detector)
        if not face_paths:
            raise RuntimeError("No faces detected in the video")
        if output_zip is None:
            video_name = get_filename_only(video_path)
            output_zip = os.path.join(os.path.dirname(video_path), f"{video_name}_faces.zip")
        with zipfile.ZipFile(output_zip, 'w') as zf:
            for face in face_paths:
                zf.write(face, arcname=os.path.basename(face))
        print(f"✅ Faces saved to: {output_zip}")
        return output_zip

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract faces from video using OpenCV (no TensorFlow)")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output ZIP file path")
    parser.add_argument("-r", "--sample-rate", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("-d", "--detector", choices=['dnn', 'haar'], default='dnn', help="Face detector: dnn (more accurate) or haar (faster)")
    args = parser.parse_args()
    try:
        process_video(args.video, args.output, args.sample_rate, args.detector)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)