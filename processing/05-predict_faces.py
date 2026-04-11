import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def validate_faces_folder(faces_folder: str) -> tuple:
    if not os.path.exists(faces_folder):
        return False, (
            f"Folder not found: '{faces_folder}'\n"
            f"  Make sure face detection ran first and saved faces to this folder.\n"
            f"  Expected path: {os.path.abspath(faces_folder)}"
        )

    if not os.path.isdir(faces_folder):
        return False, f"Path exists but is not a folder: '{faces_folder}'"

    all_files    = os.listdir(faces_folder)
    image_files  = [f for f in all_files if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not all_files:
        return False, (
            f"Folder '{faces_folder}' is empty.\n"
            f"  No files were saved by the face detector.\n"
            f"  Check that your face detection step ran correctly."
        )

    if not image_files:
        return False, (
            f"Folder '{faces_folder}' has {len(all_files)} file(s) but none are images.\n"
            f"  Found: {all_files[:5]}{'...' if len(all_files) > 5 else ''}\n"
            f"  Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return True, ""


def predict_faces(faces_folder: str, model_path: str, target_size: tuple = (224, 224)) -> tuple:
    """
    Predict authenticity for all face images in a folder.

    Args:
        faces_folder: Path to folder containing cropped face images.
        model_path:   Path to the trained Keras model (.keras or .h5).
        target_size:  Image resize dimensions expected by the model.

    Returns:
        (authenticity_percentage, num_faces, list_of_predictions)

    Raises:
        FileNotFoundError: If model or faces folder is missing.
        ValueError:        If no valid face images are found.
    """
    # ── Validate model ────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'\n"
            f"  Expected at: {os.path.abspath(model_path)}\n"
            f"  Train the model first or update model_path."
        )

    # ── Validate faces folder ─────────────────────────────────────────────────
    is_valid, error_msg = validate_faces_folder(faces_folder)
    if not is_valid:
        raise ValueError(error_msg)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # ── Run predictions ───────────────────────────────────────────────────────
    image_files = [
        f for f in os.listdir(faces_folder)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    print(f"Found {len(image_files)} face image(s) in '{faces_folder}'")

    predictions  = []
    failed_files = []

    for fname in image_files:
        img_path = os.path.join(faces_folder, fname)
        try:
            img = load_img(img_path, target_size=target_size)
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            pred = model.predict(arr, verbose=0)[0][0]   # probability of FAKE
            predictions.append((fname, float(pred)))
            label = 'FAKE' if pred > 0.5 else 'REAL'
            print(f"  {fname}: fake_prob={pred:.4f} ({label})")
        except Exception as e:
            failed_files.append((fname, str(e)))
            print(f"  [SKIP] {fname}: {e}")

    # ── Warn about failures ───────────────────────────────────────────────────
    if failed_files:
        print(f"\n[WARNING] {len(failed_files)} file(s) could not be processed:")
        for fname, err in failed_files:
            print(f"   - {fname}: {err}")

    if not predictions:
        raise ValueError(
            f"No faces could be processed from '{faces_folder}'.\n"
            f"  All {len(image_files)} image(s) failed during prediction.\n"
            f"  Check image integrity or model input requirements."
        )

    # ── Compute result ────────────────────────────────────────────────────────
    probs         = [p for _, p in predictions]
    avg_fake_prob = np.mean(probs)
    authenticity  = (1 - avg_fake_prob) * 100

    return authenticity, len(predictions), predictions


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 05-predict_faces.py <faces_folder> [model_path]")
        print("  faces_folder : folder with cropped face images (e.g. ./faces)")
        print("  model_path   : optional override for model location")
        sys.exit(1)

    faces_folder = sys.argv[1]
    model_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(os.path.dirname(__file__), '..', 'tmp_checkpoint', 'best_model.keras')
    )

    try:
        score, count, preds = predict_faces(faces_folder, model_path)

        print(f"\n{'-'*40}")
        print(f"  Faces processed : {count}")
        print(f"  Authenticity    : {score:.2f}%")
        print(f"  Verdict         : {'Likely REAL' if score >= 50 else 'Likely FAKE'}")
        print(f"{'-'*40}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found:\n{e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Face folder issue:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)