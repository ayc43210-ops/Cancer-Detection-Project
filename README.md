# DermAI: Skin Cancer Detection

DermAI is a clinical decision-support prototype that classifies skin lesion images into one of seven diagnostic categories using a deep learning model (transfer learning on MobileNetV2, trained on the HAM10000 dataset). It ships with two ways to use the model: a **Streamlit app** for interactive use and a **FastAPI backend** that the React dashboard calls.

**Frontend:** [ayc43210-ops/dermai-dashboard](https://github.com/ayc43210-ops/dermai-dashboard) (Vite + React + TypeScript). It sends images to this backend's `/predict` endpoint.

> **Disclaimer:** This is an educational / research project. It is **not** a medical device and must not be used for real diagnosis or treatment decisions. Always consult a qualified dermatologist.

## Features

- Upload a lesion image (`jpg`, `jpeg`, `png`) and get a predicted class with a confidence score
- Per-class probability bar chart
- Risk level and lesion type (benign / pre-malignant / malignant) for each prediction
- Downloadable plain-text clinical report
- Session history log (Streamlit app)
- REST API (`POST /predict`) with CORS enabled, used by the [DermAI dashboard](https://github.com/ayc43210-ops/dermai-dashboard)
- Predictions saved to `backend/patient_data.json` by the API

## Classes

| Class | Type | Risk |
|---|---|---|
| Actinic Keratoses | Pre-Malignant | High |
| Basal Cell Carcinoma | Malignant | Critical |
| Benign Keratosis | Benign | Low |
| Dermatofibroma | Benign | Low |
| Melanoma | Malignant | Critical |
| Nevus | Benign | Low |
| Vascular Lesions | Benign | Low |

## Project Structure

```
Cancer-Detection-Project/
├── backend/
│   ├── app.py                 # Streamlit app (DermAI dashboard)
│   ├── main.py                # FastAPI service exposing /predict
│   ├── requirements.txt       # Python dependencies
│   ├── patient_data.json      # Prediction log written by the API
│   └── models/
│       └── skin_cancer_model.h5   # Trained Keras model used for inference
├── frontend/                  # Empty placeholder; the real frontend is the separate dermai-dashboard repo
├── notebooks/
│   ├── training.py            # PyTorch transfer-learning setup (work in progress)
│   └── exploration.py         # Placeholder
├── scripts/
│   └── download_data.py       # Placeholder
└── README.md
```

## Model

- **Architecture:** MobileNetV2 backbone with a custom classification head (transfer learning)
- **Input:** RGB image resized to 224 × 224, pixel values scaled to `[0, 1]`
- **Output:** 7-class softmax
- **Dataset:** HAM10000
- **Reported validation accuracy:** 93.25% (figure taken from the app's own description; the evaluation code is not included in this repository)

## Getting Started

### Prerequisites

- Python 3.9+
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/ayc43210-ops/Cancer-Detection-Project.git
cd Cancer-Detection-Project/backend
```

All commands below must be run from the `backend/` folder, because the model is loaded from the relative path `models/skin_cancer_model.h5`.

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install tensorflow fastapi uvicorn python-multipart
```

The inference code uses TensorFlow/Keras, FastAPI and Uvicorn, which are not yet listed in `requirements.txt`, so they are installed separately above.

### 3a. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501), upload a lesion image, and view the result.

### 3b. Run the API

```bash
uvicorn main:app --reload
```

The API is served at http://127.0.0.1:8000, with interactive docs at http://127.0.0.1:8000/docs.

### 4. Run the dashboard (optional)

In a second terminal:

```bash
git clone https://github.com/ayc43210-ops/dermai-dashboard.git
cd dermai-dashboard
cp .env.example .env        # VITE_API_URL defaults to http://127.0.0.1:8000
npm install
npm run dev
```

Open the URL Vite prints. The ONLINE indicator in the top bar turns green when it can reach this API.

## API Reference

### `GET /`

Health check.

```json
{ "status": "Online", "model": "MobileNetV2 (93.25% Acc)" }
```

### `POST /predict`

Upload an image as `multipart/form-data` with the field name `file`.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@lesion.jpg"
```

Successful response:

```json
{
  "success": true,
  "classification": "Melanoma",
  "confidence": "98.18%",
  "risk_level": "Critical",
  "type": "Malignant"
}
```

On failure, the API returns `{ "success": false, "error": "<message>" }`.

## Known Limitations / Roadmap

- The `frontend/` folder in this repo is empty and can be deleted; the UI lives in the [dermai-dashboard](https://github.com/ayc43210-ops/dermai-dashboard) repo.
- `patient_data.json` is a simple local log of predictions; the dashboard does not read it, and its session history resets on refresh.
- `notebooks/training.py` is a PyTorch setup script (dataset class, augmentation, model head, optimizer) without a training loop, while the deployed model is a Keras `.h5` file, so the training pipeline is not reproducible from this repo yet.
- `notebooks/exploration.py` and `scripts/download_data.py` are empty.
- `requirements.txt` lists PyTorch but not the TensorFlow, FastAPI and Uvicorn packages the backend actually uses.
- `backend/models/skin_model.h5` is an empty file; only `skin_cancer_model.h5` is a working model.
- The model has not been validated for clinical use, and HAM10000 is class-imbalanced and not representative of all skin types, so results may not generalize.
- `allow_origins=["*"]` in `main.py` should be restricted to the frontend's URL before any deployment.

## Acknowledgements

- [HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161) (Tschandl et al., 2018)
- MobileNetV2 (Sandler et al., 2018)
