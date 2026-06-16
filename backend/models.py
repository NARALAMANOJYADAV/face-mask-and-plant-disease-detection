from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base

class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    prediction_label = Column(String)  # "With Mask" or "Without Mask"
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
