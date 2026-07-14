import importlib.util
import sys
import tiktoken
from datetime import date
from huggingface_hub import hf_hub_download

CACHE_DIR = "/scratch/jschwab/hf"
_AVAILABLE_YEARS = range(1999, 2025)

# Model-type registry: maps model_type → (repo pattern, class filename, class name)
_MODEL_REGISTRY = {
    "base": (
        "manelalab/chrono-gpt-v1-{year}1231",
        "ChronoGPT_inference.py",
        "ChronoGPT",
    ),
    "instruct": (
        "manelalab/chrono-gpt-instruct-v1-{year}1231",
        "ChronoGPT_instruct.py",
        "ChronoGPT",
    ),
}


def load_model(target_date: date, model_type: str = "instruct"):
    """Load a ChronoGPT model snapshot for the given year.

    Args:
        target_date: datetime.date whose year selects the snapshot (1999–2024).
        model_type: "base" or "instruct".

    Returns:
        (tokenizer, model) tuple.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type {model_type!r}. Choose from: {list(_MODEL_REGISTRY)}")

    year = target_date.year
    if year not in _AVAILABLE_YEARS:
        raise ValueError(
            f"No ChronoGPT snapshot for year {year}. "
            f"Available years: {_AVAILABLE_YEARS.start}–{_AVAILABLE_YEARS.stop - 1}."
        )

    repo_pattern, class_file, class_name = _MODEL_REGISTRY[model_type]
    model_id = repo_pattern.format(year=year)
    print(f"  Loading model: {model_id}")

    # Download the custom model class from the repo and import it dynamically
    script_path = hf_hub_download(
        repo_id=model_id,
        filename=class_file,
        cache_dir=CACHE_DIR,
    )
    module_name = f"chronogpt_{model_type}_{year}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # register so torch.compile can resolve it
    spec.loader.exec_module(module)

    model = getattr(module, class_name).from_pretrained(model_id, cache_dir=CACHE_DIR)
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer, model
