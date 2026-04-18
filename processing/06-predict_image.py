import os
import sys
import cv2
import numpy as np
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

PROTO_PATH = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
MODEL_PATH_DNN = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000_fp16.caffemodel")

def download_model(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            raise RuntimeError(f"Failed to download model file '{os.path.basename(dest)}': {e}")

def get_face_detector():
    download_model(
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt",
        PROTO_PATH
    )
    download_model(
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        MODEL_PATH_DNN
    )
    try:
        net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH_DNN)
    except Exception as e:
        raise RuntimeError(f"Failed to load face detector model: {e}")

    def detect(image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        dets = net.forward()
        faces = []
        for i in range(dets.shape[2]):
            conf = dets[0, 0, i, 2]
            if conf > 0.5:
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype("int")
                faces.append({'box': (x, y, x2 - x, y2 - y), 'confidence': float(conf)})
        return faces
    return detect


def predict_image(image_path: str, model_path: str, target_size: tuple = (224, 224)) -> dict:
    """
    Detect faces in a single image and predict deepfake probability.

    Returns dict with:
        authenticity  : float (0-100), higher = more real
        verdict       : str
        face_count    : int
        predictions   : list of per-face results
        no_face       : bool — True if no face detected, runs whole-image fallback
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── Detect faces ──────────────────────────────────────────────────────────
    detector = get_face_detector()
    faces    = detector(rgb)
    print(f"Detected {len(faces)} face(s) in image")

    predictions = []
    no_face     = False

    if faces:
        # Run model on each detected face crop
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            margin_x = int(w * 0.3)
            margin_y = int(h * 0.3)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(rgb.shape[1], x + w + margin_x)
            y2 = min(rgb.shape[0], y + h + margin_y)
            crop = rgb[y1:y2, x1:x2]

            try:
                if crop.size == 0:
                    raise ValueError("Empty crop")
                resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                arr = img_to_array(resized)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_input(arr)
                pred = model.predict(arr, verbose=0)[0][0]
                predictions.append({'face': i, 'fake_prob': float(pred)})
                label = 'FAKE' if pred > 0.5 else 'REAL'
                print(f"  Face {i}: fake_prob={pred:.4f} ({label})")
            except Exception as e:
                print(f"  [SKIP] Face {i}: {e}")

    else:
        # No face detected — run model on full image as fallback
        print("  No face detected — running model on full image as fallback")
        no_face = True
        try:
            img = load_img(image_path, target_size=target_size)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            pred = model.predict(arr, verbose=0)[0][0]
            predictions.append({'face': 'full_image', 'fake_prob': float(pred)})
            label = 'FAKE' if pred > 0.5 else 'REAL'
            print(f"  Full image: fake_prob={pred:.4f} ({label})")
        except Exception as e:
            raise ValueError(f"Prediction failed on full image: {e}")

    if not predictions:
        raise ValueError("No predictions could be made from this image.")

    avg_fake  = np.mean([p['fake_prob'] for p in predictions])
    auth      = round((1 - avg_fake) * 100, 2)
    verdict   = 'Likely REAL' if auth >= 50 else 'Likely FAKE'

    print(f"\n{'-'*40}")
    print(f"  Faces processed : {len(predictions)}")
    print(f"  Authenticity    : {auth:.2f}%")
    print(f"  Verdict         : {verdict}")
    print(f"{'-'*40}")

    return {
        'authenticity': auth,
        'verdict':      verdict,
        'face_count':   len(predictions),
        'predictions':  predictions,
        'no_face':      no_face
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 06-predict_image.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = (
        sys.argv[2] if len(sys.argv) > 2
        else os.path.join(os.path.dirname(__file__), '..', 'tmp_checkpoint', 'best_model.keras')
    )

    try:
        result = predict_image(image_path, model_path)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
