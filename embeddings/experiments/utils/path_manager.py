import os
import load_dotenv

load_dotenv.load_dotenv()  # Load environment variables from .env file
_EMBEDDINGS_BASE_DIR = os.environ["EMBEDDINGS_DIR"]
def get_embeddings_path(model_name:str) -> str:
    
    base_dir = _EMBEDDINGS_BASE_DIR
    if "mistral" in model_name:
        model_dir = os.path.join(base_dir, model_name, "mistral-7b-instruct_ew.pkl")
    else:
        model_dir = os.path.join(base_dir, model_name, "embeddings_monthly.pkl")
    
    if "dummy" in model_name:
        return "dummy"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory for model '{model_name}' not found at {model_dir}")
    
    return model_dir