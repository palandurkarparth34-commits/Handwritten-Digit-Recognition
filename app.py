"""HR: MNIST-trained handwritten digit recognition."""
from __future__ import annotations

import base64
import binascii
import os
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from urllib.request import Request, urlopen

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "hr-mnist-v4.joblib"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MNIST_CACHE = ROOT / ".mnist-cache"
MNIST_FILE = MNIST_CACHE / "mnist.npz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 7 * 1024 * 1024
_model = None
_model_lock = Lock()
_status_lock = Lock()
_model_status = {
    "state": "starting",
    "stage": "Checking for a saved model",
    "progress": 3,
    "message": "Checking whether the model is already installed…",
    "ready": False,
    "error": None,
}


def set_model_status(state: str, stage: str, progress: int, message: str, error: str | None = None):
    """Publish a safe, user-facing view of the model setup process."""
    with _status_lock:
        _model_status.update({
            "state": state,
            "stage": stage,
            "progress": max(0, min(100, int(progress))),
            "message": message,
            "ready": state == "ready",
            "error": error,
        })


def get_model_status():
    with _status_lock:
        return dict(_model_status)


def get_model():
    """Load a cached model or download MNIST and train a small neural network."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        if MODEL_PATH.is_file():
            set_model_status("loading", "Loading saved model", 92, "Loading your saved model…")
            try:
                _model = joblib.load(MODEL_PATH)
            except Exception:
                # A damaged or outdated cache should not permanently block setup.
                app.logger.exception("Cached model could not be loaded; rebuilding it")
                MODEL_PATH.unlink(missing_ok=True)
            else:
                set_model_status("ready", "Ready", 100, "The recognition model is ready.")
                return _model

        try:
            MNIST_CACHE.mkdir(parents=True, exist_ok=True)
            if not MNIST_FILE.is_file() or MNIST_FILE.stat().st_size < 1_000_000:
                set_model_status(
                    "downloading", "Downloading MNIST dataset", 12,
                    "Downloading about 11 MB of training data from Google-hosted MNIST. This is separate from your browser connection.",
                )
                request = Request(MNIST_URL, headers={"User-Agent": "DigitStudio/6.0"})
                temporary_file = MNIST_FILE.with_suffix(".download")
                with urlopen(request, timeout=30) as response, open(temporary_file, "wb") as output:
                    total = int(response.headers.get("Content-Length", "0") or 0)
                    downloaded = 0
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            progress = 12 + int(24 * downloaded / total)
                            set_model_status(
                                "downloading", "Downloading MNIST dataset", min(progress, 36),
                                f"Downloading MNIST training data · {downloaded / (1024 * 1024):.1f} MB" + (f" of {total / (1024 * 1024):.1f} MB…" if total else " received…"),
                            )
                if temporary_file.stat().st_size < 1_000_000:
                    raise RuntimeError("The downloaded dataset file is unexpectedly small.")
                temporary_file.replace(MNIST_FILE)
            else:
                set_model_status("loading", "Loading cached MNIST data", 36, "Using the MNIST dataset already downloaded on this PC…")
            set_model_status("training", "Preparing training data", 40, "MNIST is ready. Preparing 60,000 training images…")
            with np.load(MNIST_FILE) as dataset:
                x = np.asarray(dataset["x_train"], dtype=np.float32).reshape(-1, 784) / 255.0
                y = np.asarray(dataset["y_train"], dtype=np.int64)
        except Exception as exc:
            app.logger.exception("Could not download or load MNIST from %s", MNIST_URL)
            message = (
                "The MNIST download failed. Your browser can have internet while Python cannot reach this data host. "
                "Restart the app and check the PowerShell/terminal log for the exact network error. "
                f"Technical detail: {type(exc).__name__}: {exc}"
            )
            set_model_status("error", "Download failed", 12, message, message)
            raise RuntimeError(message) from exc

        set_model_status("training", "Preparing training data", 42, "MNIST downloaded. Preparing 60,000 training images…")
        try:
            classifier = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=1,
                warm_start=True,
                random_state=42,
                verbose=False,
            )
            # partial_fit lets the UI report real epoch progress instead of
            # pretending the dataset download has a known percentage.
            epochs = 15
            for epoch in range(1, epochs + 1):
                if epoch == 1:
                    classifier.partial_fit(x, y, classes=np.arange(10))
                else:
                    classifier.partial_fit(x, y)
                progress = 44 + round((epoch / epochs) * 50)
                set_model_status(
                    "training", "Training neural network", progress,
                    f"Training on 60,000 MNIST images · epoch {epoch} of {epochs}…",
                )

            set_model_status("saving", "Saving model", 96, "Saving the trained model for future launches…")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(classifier, MODEL_PATH, compress=3)
            _model = classifier
            app.logger.info("MNIST model trained; epochs=%s", epochs)
            set_model_status("ready", "Ready", 100, "The recognition model is ready.")
            return _model
        except Exception as exc:
            app.logger.exception("MNIST model training or saving failed")
            message = "The model setup failed. Check that your PC has enough free memory, then restart the app."
            set_model_status("error", "Model setup failed", 95, message, message)
            raise RuntimeError(message) from exc


def image_to_features(image_bytes: bytes) -> np.ndarray:
    """Preprocess a drawing/photo into a 28x28 MNIST-like image, flattened 0..1."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
    except Exception as exc:
        raise ValueError("That file could not be opened as an image.") from exc

    # Composite transparent pixels over white before detecting the background.
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(white, image).convert("L")
    if gray.width < 2 or gray.height < 2:
        raise ValueError("The image is too small to recognise.")
    if gray.width * gray.height > 20_000_000:
        raise ValueError("The image dimensions are too large. Please use a smaller image.")

    pixels = np.asarray(gray, dtype=np.float32)
    border = np.concatenate((pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]))
    background_is_light = float(np.median(border)) >= 127.5
    ink = 255.0 - pixels if background_is_light else pixels.copy()
    peak = float(ink.max())
    if peak < 18:
        raise ValueError("No digit was found. Draw or upload a darker digit.")
    ink[ink < max(10.0, peak * 0.10)] = 0.0
    coords = np.argwhere(ink > 0)
    if coords.size == 0:
        raise ValueError("No digit was found. Draw or upload a darker digit.")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    crop = ink[y0:y1, x0:x1]
    height, width = crop.shape

    # MNIST digits typically occupy roughly a 20x20 box inside a 28x28 frame.
    scale = 20.0 / max(height, width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    resized = Image.fromarray(np.uint8(np.clip(crop, 0, 255))).resize(
        (resized_w, resized_h), Image.Resampling.LANCZOS
    )
    normalized = np.zeros((28, 28), dtype=np.float32)
    top = (28 - resized_h) // 2
    left = (28 - resized_w) // 2
    normalized[top:top + resized_h, left:left + resized_w] = np.asarray(resized, dtype=np.float32) / 255.0

    # MNIST digits are approximately centre-of-mass aligned; match that convention.
    center_y, center_x = ndimage.center_of_mass(normalized)
    if np.isfinite(center_y) and np.isfinite(center_x):
        normalized = ndimage.shift(
            normalized,
            (13.5 - center_y, 13.5 - center_x),
            order=1,
            mode="constant",
            cval=0.0,
        )
    return np.clip(normalized, 0.0, 1.0).reshape(1, 784).astype(np.float32)


def decode_image_payload(value: object) -> bytes:
    """Validate a base64 or data-URL image payload and return raw bytes."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Please draw a digit or choose an image first.")
    encoded = value.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or not header.startswith("data:image/") or ";base64" not in header:
            raise ValueError("Please provide a valid base64 image.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The image data is not valid base64.") from exc
    if not raw:
        raise ValueError("The image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Please use an image under 5 MB.")
    return raw


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None, "model": "MNIST MLP"})


@app.get("/api/status")
def model_status():
    """Return live model-download/training progress for the startup screen."""
    response = jsonify(get_model_status())
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be JSON."}), 400
    try:
        image_bytes = decode_image_payload(payload.get("image"))
        features = image_to_features(image_bytes)
        model = get_model()
        probabilities = model.predict_proba(features)[0]
        classes = [int(value) for value in model.classes_]
        if len(probabilities) != len(classes) or not len(probabilities):
            raise RuntimeError("The model returned an invalid probability array.")
        best_index = int(np.argmax(probabilities))
        digit = classes[best_index]
        probability_map = {
            str(label): round(float(score) * 100.0, 1)
            for label, score in zip(classes, probabilities)
        }
        return jsonify({
            "prediction": digit,
            "confidence": round(float(probabilities[best_index]) * 100.0, 1),
            "probabilities": {str(number): probability_map.get(str(number), 0.0) for number in range(10)},
            "empty": False,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        app.logger.exception("Digit recognition setup failed")
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Digit recognition failed")
        return jsonify({"error": "The model could not analyse this image. Please try again."}), 500


@app.errorhandler(413)
def image_too_large(_error):
    return jsonify({"error": "Image is too large. Please use an image under 5 MB."}), 413


def warm_model_in_background():
    """Download/train at startup so the website can show live setup progress."""
    try:
        get_model()
    except Exception:
        # The error details are already reflected in /api/status and the log.
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"HR is running at http://127.0.0.1:{port}")
    print("Starting model setup in the background; the website will show its progress.")
    Thread(target=warm_model_in_background, name="mnist-model-setup", daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)
