from pathlib import Path
import os

# --------------------------------------------------
# BASE DIRECTORY (Project Root)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# DATA PATHS
# --------------------------------------------------
# Default: project/data/inventory.db
# Can be overridden via environment variable: DB_PATH
DATA_PATH = Path(
    os.getenv("DB_PATH", BASE_DIR / "data" / "inventory.db")
)


# --------------------------------------------------
# MODEL DIRECTORY
# --------------------------------------------------
# Default: project/models/
# Can be overridden via environment variable: MODEL_DIR
MODEL_DIR = Path(
    os.getenv("MODEL_DIR", BASE_DIR / "models")
)

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# MODEL FILE PATHS
# --------------------------------------------------
FREIGHT_MODEL_PATH = MODEL_DIR / "freight_model.pkl"
INVOICE_MODEL_PATH = MODEL_DIR / "invoice_model.pkl"


# --------------------------------------------------
# OPTIONAL: DEBUG / VALIDATION
# --------------------------------------------------
if not DATA_PATH.exists():
    print(f"[WARNING] Data file not found at: {DATA_PATH}")

if not MODEL_DIR.exists():
    print(f"[WARNING] Model directory not found at: {MODEL_DIR}")