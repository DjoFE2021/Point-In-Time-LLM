"""Resolve a model's embeddings pickle under EMBEDDINGS_DIR (read from env or .env)."""
import os
from dotenv import load_dotenv

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_FILE)  # embeddings/experiments/.env; already-set variables take precedence
_EMBEDDINGS_BASE_DIR = os.environ.get("EMBEDDINGS_DIR")
if not _EMBEDDINGS_BASE_DIR:
    raise RuntimeError(f"EMBEDDINGS_DIR is not set: add it to {_ENV_FILE} (see .env.example)")


def get_embeddings_path(model_name:str) -> str:
    """Return EMBEDDINGS_DIR/<model_name>/embeddings_monthly.pkl (output of the embedding jobs)."""

    base_dir = _EMBEDDINGS_BASE_DIR
    model_dir = os.path.join(base_dir, model_name, "embeddings_monthly.pkl")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory for model '{model_name}' not found at {model_dir}")

    return model_dir
