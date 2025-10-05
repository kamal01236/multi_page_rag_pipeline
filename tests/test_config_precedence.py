import os
from importlib import reload

import src.config as config_mod


def test_env_overrides_config_file(monkeypatch, tmp_path):
    # create a temp config file with default_backend: tfidf
    cfg = tmp_path / "config.yaml"
    cfg.write_text("default_backend: huggingface\nforce_ollama: false\n")
    monkeypatch.setenv('RAG_CONFIG_FILE', str(cfg))
    monkeypatch.setenv('RAG_DEFAULT_BACKEND', 'openai')
    # reload config module to pick up env and file
    reload(config_mod)
    cfg_obj = config_mod.CONFIG
    assert cfg_obj.default_backend == 'openai'
