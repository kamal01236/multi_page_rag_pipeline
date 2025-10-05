import os
from src.rag_pipeline import detect_backend


def test_detect_backend_openai(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    assert detect_backend() == 'openai'


def test_detect_backend_any(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    # The environment may or may not have Ollama; ensure detect_backend returns a valid token
    val = detect_backend()
    assert val in {'ollama', 'openai', 'huggingface', 'tfidf', 'none'}
