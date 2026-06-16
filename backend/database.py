import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# On Render (and other cloud hosts) the working directory is read-only.
# Use /tmp for SQLite so the DB is writable. Note: /tmp is ephemeral on Render free tier.
# For persistent storage upgrade to a Render PostgreSQL database and swap the URL below.
_DB_DIR  = "/tmp" if os.environ.get("RENDER") else os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_DB_DIR, "masks.db")

SQLALCHEMY_DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
