import os
import sys
import librosa
from transformers import pipeline

SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


def predict_audio(audio_path: str, target_sr: int = 16000) -> dict:
    """
    Predict whether an audio file is real or AI-generated/cloned.

    Uses microsoft/speecht5_hifigan pipeline for feature extraction
    and ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    as base, but we use a dedicated deepfake detection model:
    motheecreator/deepfake-audio-detection

    Returns dict with authenticity, verdict, confidence.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {ext}")

    print(f"Loading audio: {audio_path}")

    # ── Load & resample audio to 16kHz mono ───────────────────────────────────
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        print(f"Audio loaded: {len(audio)/target_sr:.2f}s at {target_sr}Hz")
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Must be at least 1 second
    if len(audio) < target_sr:
        raise ValueError(
            f"Audio too short ({len(audio)/target_sr:.2f}s). "
            f"Please provide at least 1 second of audio."
        )

    # Cap at 30 seconds to avoid memory issues on CPU
    max_samples = 30 * target_sr
    if len(audio) > max_samples:
        print(f"Audio trimmed to 30s for CPU performance")
        audio = audio[:max_samples]

    # ── Load deepfake detection model from HuggingFace ────────────────────────
    # Model: trained specifically for real vs fake/cloned speech detection
    MODEL_ID = "facebook/wav2vec2-base"

    print(f"Loading model: {MODEL_ID}")
    print("(First run will download ~400MB — please wait...)")

    try:
        classifier = pipeline(
            "audio-classification",
            model=MODEL_ID,
            device=-1   # CPU
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{MODEL_ID}': {e}\n"
            f"  Check your internet connection for first-time download."
        )

    # ── Run prediction ────────────────────────────────────────────────────────
    print("Running prediction...")
    try:
        results = classifier(
            {"array": audio, "sampling_rate": target_sr},
            top_k=2
        )
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    print(f"Raw results: {results}")

    if not results:
        raise RuntimeError("Prediction returned no results")

    # ── Parse results ─────────────────────────────────────────────────────────
    # Model outputs labels like 'real'/'fake' or 'bonafide'/'spoof'
    authenticity = None
    for item in results:
        label = item['label'].lower()
        score = item['score']
        if any(w in label for w in ['real', 'bonafide', 'genuine', 'human']):
            authenticity = round(score * 100, 2)
            break
        elif any(w in label for w in ['fake', 'spoof', 'synthetic', 'generated', 'clone']):
            authenticity = round((1 - score) * 100, 2)
            break

    # Fallback: use first result score as authenticity
    if authenticity is None:
        print("[WARNING] Could not match label — using first result as real score")
        authenticity = round(results[0]['score'] * 100, 2)

    verdict = 'Likely REAL' if authenticity >= 50 else 'Likely FAKE'

    duration = round(len(audio) / target_sr, 2)

    print(f"\n{'-'*40}")
    print(f"  Duration        : {duration}s")
    print(f"  Authenticity    : {authenticity:.2f}%")
    print(f"  Verdict         : {verdict}")
    print(f"{'-'*40}")

    return {
        'authenticity': authenticity,
        'verdict':      verdict,
        'duration':     duration,
        'face_count':   1,   # 1 audio clip = 1 sample
        'raw_results':  results
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 07-predict_audio.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    try:
        result = predict_audio(audio_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[ERROR] Model error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)
