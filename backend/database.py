import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# DATABASE_URL must be set as an environment variable.
# On Render: use the Internal Database URL (faster, no egress fees).
# Locally:   set it in your .env file.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# SQLAlchemy requires 'postgresql://' not 'postgres://' (older Render URLs use postgres://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Add it to your .env file locally or to Render environment variables."
    )

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
