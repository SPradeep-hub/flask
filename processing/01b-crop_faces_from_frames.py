import os
import cv2
import numpy as np

def get_face_detector(method='dnn'):
    """Return a face detector function (DNN or Haar)."""
    if method == 'dnn':
        # Paths to model files (will be downloaded automatically if missing)
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
                if confidence > 0.5:  # threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    faces.append({'box': (x, y, x2-x, y2-y), 'confidence': confidence})
            return faces
        return detect_faces
    else:  # Haar cascade fallback
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        def detect_faces(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return [{'box': (x, y, w, h), 'confidence': 1.0} for (x, y, w, h) in rects]
        return detect_faces

def crop_faces_from_frames(frame_folder, output_folder, margin_ratio=0.3, min_confidence=0.5, detector_method='dnn'):
    """
    Process all images in frame_folder, detect faces, crop them, and save to output_folder.
    Returns list of saved face file paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    detector = get_face_detector(detector_method)
    face_paths = []
    for frame_file in os.listdir(frame_folder):
        if not frame_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        frame_path = os.path.join(frame_folder, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector(rgb)
        for i, res in enumerate(results):
            if res['confidence'] < min_confidence:
                continue
            x, y, w, h = res['box']
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(rgb.shape[1], x + w + margin_x)
            y2 = min(rgb.shape[0], y + h + margin_y)
            crop = rgb[y1:y2, x1:x2]
            base_name = os.path.splitext(frame_file)[0]
            out_name = f"{base_name}_face{i:02d}.png"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            face_paths.append(out_path)
    return face_paths

if __name__ == "__main__":
    # CLI usage: python crop_faces.py <frames_folder> <output_folder> [detector]
    import sys
    if len(sys.argv) < 3:
        print("Usage: python 01b-crop_faces_from_frames.py <frames_folder> <output_folder> [detector]")
        sys.exit(1)
    frames_folder = sys.argv[1]
    output_folder = sys.argv[2]
    detector_method = sys.argv[3] if len(sys.argv) > 3 else 'dnn'
    crop_faces_from_frames(frames_folder, output_folder, detector_method=detector_method)