from dataclasses import dataclass
import os
import json
from typing import Optional

try:
    import yaml  # optional; YAML support if PyYAML is installed
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False


@dataclass
class Config:
    # Backend selection: 'auto' | 'openai' | 'ollama' | 'huggingface' | 'tfidf'
    default_backend: str = "auto"
    force_ollama: bool = False
    # Explicit control for embedding and LLM backends
    emb_backend: Optional[str] = None
    llm_backend: Optional[str] = None

    # Standardized env names
    openai_api_key: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_token: Optional[str] = None
    huggingfacehub_api_token: Optional[str] = None

    # Optional config file path
    config_file: Optional[str] = None


def _read_config_file(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        if HAVE_YAML and path.lower().endswith((".yml", ".yaml")):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


def load_config() -> Config:
    cfg_path = os.getenv("RAG_CONFIG_FILE") or "./config/config.yaml"
    if not os.path.exists(cfg_path):
        alt = "./config.yaml"
        if os.path.exists(alt):
            cfg_path = alt
    file_conf = _read_config_file(cfg_path) if cfg_path else {}

    def env_or_file(key: str, file_key: Optional[str] = None):
        return os.getenv(key) or file_conf.get(file_key or key.lower())

    default_backend = env_or_file("RAG_DEFAULT_BACKEND", "default_backend") or "auto"
    force_ollama = os.getenv("FORCE_OLLAMA", "").lower() in {"1", "true", "yes"} or bool(file_conf.get("force_ollama"))

    ollama_url = env_or_file("OLLAMA_URL") or env_or_file("OLLAMA_BASE_URL") or file_conf.get("ollama_url")
    return Config(
        default_backend=default_backend,
        force_ollama=force_ollama,
        emb_backend=env_or_file("RAG_EMB_BACKEND") or file_conf.get("emb_backend"),
        llm_backend=env_or_file("RAG_LLM_BACKEND") or file_conf.get("llm_backend"),
        openai_api_key=env_or_file("OPENAI_API_KEY"),
        ollama_url=ollama_url,
        ollama_token=env_or_file("OLLAMA_TOKEN"),
        huggingfacehub_api_token=env_or_file("HUGGINGFACEHUB_API_TOKEN"),
        config_file=cfg_path if os.path.exists(cfg_path) else None,
    )


# Load at import time for convenience
CONFIG = load_config()
