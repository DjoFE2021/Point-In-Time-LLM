import os
def get_embeddings_path(model_name:str) -> str:
    
    base_dir = "/srv/datasets/DOW_JONES_NEWSWIRE/embeddings"
    model_dir = os.path.join(base_dir, model_name, "embeddings_monthly.pkl")
    
    if "dummy" in model_name:
        return "dummy"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory for model '{model_name}' not found at {model_dir}")
    
    return model_dir