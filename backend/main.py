import os
from dotenv import load_dotenv
load_dotenv()  # loads .env file into os.environ
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import cv2
import numpy as np
import base64
import sys

from .database import engine, Base, get_db
from .models import DetectionHistory

# Init DB
Base.metadata.create_all(bind=engine)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..', 'frontend'))

sys.path.append(os.path.abspath(os.path.join(BACKEND_DIR, '..')))
from src.predict import FaceMaskPredictor

app = FastAPI()

# Serve frontend HTML files
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

@app.on_event("startup")
def load_model():
    global predictor
    model_path = os.path.abspath(os.path.join(BACKEND_DIR, '..', 'models', 'best_model.h5'))
    predictor = FaceMaskPredictor(model_path=model_path)


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run Prediction
    processed_image, detections = predictor.predict(image)

    # Encode image back to base64
    _, buffer = cv2.imencode('.jpg', processed_image)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    # Pick highest confidence detection to save to DB (Assume main face)
    main_detection_label = "No Detections"
    main_confidence = 0.0
    if detections:
        main_detection = max(detections, key=lambda x: x['confidence'])
        main_detection_label = main_detection['label']
        main_confidence = main_detection['confidence']

    # Log to History
    history_record = DetectionHistory(
        filename=file.filename,
        prediction_label=main_detection_label,
        confidence=main_confidence
    )
    db.add(history_record)
    db.commit()
    db.refresh(history_record)

    return {"image": img_b64, "detections": detections}

@app.get("/api/config")
def get_config():
    """Expose non-secret runtime config to the frontend."""
    return {"groq_api_key": os.environ.get("GROQ_API_KEY", "")}


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(DetectionHistory).count()
    if total == 0:
        return {"total": 0, "mask_count": 0, "no_mask_count": 0}

    mask_count    = db.query(DetectionHistory).filter(DetectionHistory.prediction_label == "With Mask").count()
    no_mask_count = db.query(DetectionHistory).filter(DetectionHistory.prediction_label == "Without Mask").count()

    return {"total": total, "mask_count": mask_count, "no_mask_count": no_mask_count}

@app.get("/api/history")
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    records = db.query(DetectionHistory).order_by(DetectionHistory.timestamp.desc()).limit(limit).all()
    return records
