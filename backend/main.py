"""GateKeeper AI incident API.

For an 8-hour demo. The reset endpoint is disabled by default; do not enable it
on a public deployment without adding proper authentication/authorization.
"""

import base64
import binascii
import os
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./incidents.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

MAX_SNAPSHOT_BASE64_CHARS = 240_000
MAX_DECODED_SNAPSHOT_BYTES = 180_000
ALLOWED_THREATS = ("WARNING (OCCLUDED)", "CRITICAL BREACH")
ALLOWED_CAMERA_MODES = ("Front-Facing", "Entrance-Wall")


class IncidentDB(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    track_id = Column(Integer, nullable=False)
    zone_name = Column(String(120), default="Restricted Server Room", nullable=False)
    threat_level = Column(String(40), nullable=False)
    camera_mode = Column(String(40), nullable=False)
    dwell_time_sec = Column(Float, nullable=False)
    # Kept in the DB for the hackathon demo. Prefer object storage + a URL at scale.
    snapshot_base64 = Column(Text, nullable=True)


Base.metadata.create_all(bind=engine)


class IncidentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int = Field(ge=0, le=2_147_483_647)
    zone_name: str = Field(default="Restricted Zone Alpha", min_length=1, max_length=120)
    threat_level: Literal["WARNING (OCCLUDED)", "CRITICAL BREACH"]
    camera_mode: Literal["Front-Facing", "Entrance-Wall"]
    dwell_time_sec: float = Field(ge=0, le=86_400, allow_inf_nan=False)
    snapshot_base64: str | None = Field(default=None, max_length=MAX_SNAPSHOT_BASE64_CHARS)

    @field_validator("snapshot_base64")
    @classmethod
    def validate_snapshot(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("snapshot_base64 must contain valid base64 data") from exc
        if len(decoded) > MAX_DECODED_SNAPSHOT_BYTES:
            raise ValueError("snapshot is too large; resize/compress it before upload")
        return value


class IncidentResponse(IncidentCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime


app = FastAPI(title="GateKeeper AI Cloud Command API", version="0.2.0")

# Comma-separated URLs, e.g. https://your-dashboard.vercel.app,http://localhost:3000
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def health_check():
    return {"status": "online", "service": "GateKeeper AI Cloud API"}


@app.post("/api/incidents", response_model=IncidentResponse, status_code=201)
def log_incident(incident: IncidentCreate, db: Session = Depends(get_db)):
    """Store one validated event received from the local edge node."""
    db_item = IncidentDB(**incident.model_dump())
    try:
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
    except Exception as exc:
        db.rollback()
        # Avoid returning database internals to the client.
        raise HTTPException(status_code=500, detail="Could not save incident") from exc
    return db_item


@app.get("/api/incidents", response_model=list[IncidentResponse])
def get_incidents(
    limit: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return (
        db.query(IncidentDB)
        .order_by(IncidentDB.timestamp.desc(), IncidentDB.id.desc())
        .limit(limit)
        .all()
    )


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(IncidentDB.id)).scalar() or 0
    critical = (
        db.query(func.count(IncidentDB.id))
        .filter(IncidentDB.threat_level == "CRITICAL BREACH")
        .scalar()
        or 0
    )
    average = db.query(func.avg(IncidentDB.dwell_time_sec)).scalar()
    return {
        "total_incidents": int(total),
        "critical_breaches": int(critical),
        "warnings": int(total - critical),
        "avg_dwell_time_sec": round(float(average), 1) if average is not None else 0.0,
    }


@app.delete("/api/incidents/clear")
def clear_incidents(db: Session = Depends(get_db)):
    """Local/private demo reset only. Disabled unless explicitly enabled.

    This is still unauthenticated, so NEVER enable it on a public deployment.
    For public deployments, implement a separate authenticated admin path instead.
    """
    if os.getenv("ENABLE_DEMO_RESET", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Not found")
    try:
        db.query(IncidentDB).delete(synchronize_session=False)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not clear incidents") from exc
    return {"message": "Demo incident logs cleared."}
