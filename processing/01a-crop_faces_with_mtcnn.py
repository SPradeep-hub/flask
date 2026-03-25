import os
import tempfile
import zipfile
from pathlib import Path

import cv2
from mtcnn import MTCNN
from flask import Flask, request, render_template_string, send_file

# Suppress TensorFlow logging and configure GPU memory growth (optional)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Initialize MTCNN detector once (reused across requests)
# Note: In a production multi-threaded environment, consider using a lock
# or process per request if thread safety becomes an issue.
detector = MTCNN()

app = Flask(__name__)

# Simple HTML upload form
UPLOAD_FORM = '''
<!doctype html>
<title>Upload Video</title>
<h1>Upload a video file to extract faces</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=video accept="video/*">
  <input type=submit value=Upload>
</form>
'''

def get_filename_only(file_path):
    """Return filename without extension."""
    return Path(file_path).stem

def extract_frames(video_path, output_dir, sample_rate=1):
    """
    Extract frames from video and save as images in output_dir.
    Returns list of frame file paths.
    """
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
    """
    Detect and crop faces from each frame, save them in faces_dir.
    Uses the global detector.
    Returns list of saved face file paths.
    """
    face_paths = []
    for frame_path in frame_paths:
        # Read image and convert to RGB (MTCNN expects RGB)
        image_bgr = cv2.imread(frame_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image_rgb)
        
        for i, result in enumerate(results):
            conf = result['confidence']
            # Skip low confidence faces if multiple faces are present
            if len(results) >= 2 and conf < min_confidence:
                print(f"Skipped low confidence face ({conf}) in {frame_path}")
                continue

            x, y, w, h = result['box']
            # Add margin
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image_rgb.shape[1], x + w + margin_x)
            y2 = min(image_rgb.shape[0], y + h + margin_y)

            crop = image_rgb[y1:y2, x1:x2]
            # Generate output filename
            base = get_filename_only(frame_path)
            face_filename = os.path.join(faces_dir, f"{base}_face{i:02d}.png")
            cv2.imwrite(face_filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            face_paths.append(face_filename)

    return face_paths

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files.get('video')
        if not video_file or video_file.filename == '':
            return "No file selected", 400

        # Create a temporary directory to store all intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded video
            video_path = os.path.join(tmpdir, video_file.filename)
            video_file.save(video_path)

            # Directory for extracted frames
            frames_dir = os.path.join(tmpdir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)

            # Extract frames (you can adjust sample_rate, e.g., 5 for every 5th frame)
            frame_paths = extract_frames(video_path, frames_dir, sample_rate=1)
            if not frame_paths:
                return "No frames could be extracted from the video", 400

            # Directory for cropped faces
            faces_dir = os.path.join(tmpdir, 'faces')
            os.makedirs(faces_dir, exist_ok=True)

            # Crop faces
            face_paths = crop_faces_from_frames(frame_paths, faces_dir)
            if not face_paths:
                return "No faces detected in the video", 200

            # Create a ZIP archive with all face images
            zip_path = os.path.join(tmpdir, 'faces.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for face in face_paths:
                    zf.write(face, arcname=os.path.basename(face))

            # Send the ZIP file to the client
            return send_file(zip_path, as_attachment=True, download_name='faces.zip')

    return render_template_string(UPLOAD_FORM)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)