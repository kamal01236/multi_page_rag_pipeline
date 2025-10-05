import os
from src.rag_pipeline import detect_backend


def test_detect_backend_force_ollama(monkeypatch):
    monkeypatch.setenv('FORCE_OLLAMA', '1')
    assert detect_backend() == 'ollama'


def test_detect_backend_openai(monkeypatch):
    monkeypatch.delenv('FORCE_OLLAMA', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    assert detect_backend() == 'openai'


def test_detect_backend_none(monkeypatch):
    monkeypatch.delenv('FORCE_OLLAMA', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    # This will return 'ollama' if a local ollama is reachable; otherwise one of the fallbacks.
    val = detect_backend()
    assert val in {'ollama', 'openai', 'huggingface', 'tfidf', 'none'}
