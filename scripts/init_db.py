#!/usr/bin/env python3
import os
import sys
import time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from alembic.config import Config
from alembic import command

# -------------------------------
# 1️⃣ Setup paths
# -------------------------------
# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Optional: dynamically import all models
import importlib
import pathlib

models_dir = pathlib.Path(PROJECT_ROOT) / "db" / "models"
for f in models_dir.glob("*.py"):
    if f.name != "__init__.py":
        importlib.import_module(f"db.models.{f.stem}")

# -------------------------------
# 2️⃣ Construct DATABASE_URL
# -------------------------------
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_db")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
print(f"DATABASE_URL: {DATABASE_URL}")
os.environ["DATABASE_URL"] = DATABASE_URL

print(f"🔗 Connecting to database at {POSTGRES_HOST}:{POSTGRES_PORT}...")

# -------------------------------
# 3️⃣ Wait for Postgres to be ready
# -------------------------------
while True:
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # ✅ wrap with text()
        break
    except OperationalError:
        print("⏳ Postgres not ready, waiting 1s...")
        time.sleep(1)

print("✅ Postgres is ready!")

# -------------------------------
# 4️⃣ Run Alembic migrations
# -------------------------------
alembic_cfg = Config(os.path.join(PROJECT_ROOT, "alembic.ini"))
alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)

# Create a new revision if there are changes (optional)
# command.revision(alembic_cfg, message="auto migration", autogenerate=True)

# Apply migrations
command.upgrade(alembic_cfg, "head")
print("✅ Alembic migrations applied!")

# -------------------------------
# 5️⃣ Optional: create tables for models not covered by Alembic
# -------------------------------
from db.core.database import Base  # Your declarative base
Base.metadata.create_all(bind=engine)
print("✅ SQLAlchemy models created (if missing).")

print("🎉 Database initialization complete!")
