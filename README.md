# HR

A responsive handwritten digit recognition app. Draw a digit in the browser or upload an image, then inspect the predicted number and its confidence distribution.

## What's included

- Redesigned midnight/aurora interface with glass-like panels, violet-to-cyan accents, staggered entrance motion, and an animated recognition loader/result reveal.
- Responsive interface with mouse, touch, and stylus drawing support.
- PNG, JPG, and WEBP upload support (5 MB raw-file limit); uploaded images are sent to the backend at their original aspect ratio instead of being stretched into the drawing canvas.
- Clear canvas and sample drawing controls.
- A Flask JSON API with input validation, stable response fields, and readable error messages.
- Lazy-loaded MNIST neural network: first launch downloads MNIST (~55 MB), trains an MLP on 60,000 training examples, and caches the trained model locally.
- No TensorFlow, API key, database, or account required. An internet connection is required for the first MNIST download.

## Run locally

Python 3.12 is recommended for the smoothest installation. The app code is compatible with recent Python releases where the listed packages provide wheels.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8080>.

The first launch downloads MNIST from OpenML (~55 MB), trains an MLP model on the 60,000-example training split, and saves it to `model/hr-mnist-v4.joblib`. Training can take several minutes. Later predictions load the cached model. The generated `model/` and `.mnist-cache/` directories are ignored by Git.

## API

`GET /api/health` returns service status.

`POST /api/predict` accepts JSON:

```json
{ "image": "data:image/png;base64,..." }
```

A successful response always has this shape:

```json
{
  "prediction": 7,
  "confidence": 92.1,
  "probabilities": {
    "0": 0.1, "1": 0.2, "2": 0.1, "3": 0.1, "4": 0.2,
    "5": 0.3, "6": 0.2, "7": 92.1, "8": 6.2, "9": 0.5
  },
  "empty": false
}
```

Invalid requests return a 4xx status and a JSON `error` field. Unexpected server errors are logged and returned as a readable 500 response. The frontend validates the response contract instead of displaying `undefined`.

## Model and accuracy note

V4 uses scikit-learn's 28×28 MNIST dataset and a multilayer perceptron (MLP), trained on the official 60,000-example training split. The app downloads the dataset on the first prediction (~55 MB) and caches both the dataset and trained model. This is a substantially more appropriate training source for handwritten digits than scikit-learn's tiny 8×8 demo dataset. No exact accuracy is claimed unless you evaluate the resulting model on a held-out test set. Confidence values are model scores, not guarantees.

The preprocessing detects light or dark backgrounds, crops the digit, preserves its aspect ratio, resizes it to a 20×20 bounding box inside a 28×28 frame, and aligns its centre of mass to resemble MNIST inputs. Very unusual handwriting, decorative writing, or photographs with clutter can still be misclassified.

## Tests

```bash
pip install pytest
pytest -q
```

## License

MIT. Retain the original repository's `LICENSE` file when applying this rebuild to the existing project.
