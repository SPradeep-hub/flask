import os
import sys
import cv2
import numpy as np
import urllib.request

# ---------- Download DNN models once ----------
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

def download_model(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest} ...")
        urllib.request.urlretrieve(url, dest)

def get_face_detector():
    download_model(
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/deploy.prototxt",
        PROTO_PATH
    )
    download_model(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        MODEL_PATH
    )
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    def detect_faces(image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:   # confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                faces.append({'box': (x, y, x2-x, y2-y), 'confidence': confidence})
        return faces
    return detect_faces

def crop_faces_from_frames(frame_folder, output_folder, margin_ratio=0.3, min_confidence=0.5):
    """
    Process all images in frame_folder, detect faces, crop them, and save to output_folder.
    Returns number of faces saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    detector = get_face_detector()
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png')
    frame_files = [f for f in os.listdir(frame_folder) if f.lower().endswith(image_extensions)]
    print(f"Found {len(frame_files)} frames in {frame_folder}")
    
    total_faces = 0
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)
        for i, face in enumerate(faces):
            if face['confidence'] < min_confidence:
                continue
            x, y, w, h = face['box']
            # Add margin
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(rgb.shape[1], x + w + margin_x)
            y2 = min(rgb.shape[0], y + h + margin_y)
            crop = rgb[y1:y2, x1:x2]
            # Output filename: frame_name_faceXX.png
            base = os.path.splitext(frame_file)[0]
            out_name = f"{base}_face{i:02d}.png"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            total_faces += 1
    print(f"Detected and saved {total_faces} faces to {output_folder}")
    return total_faces

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 01b-crop_faces_from_frames.py <frames_folder> <output_folder>")
        sys.exit(1)
    frames_folder = sys.argv[1]
    output_folder = sys.argv[2]
    crop_faces_from_frames(frames_folder, output_folder)