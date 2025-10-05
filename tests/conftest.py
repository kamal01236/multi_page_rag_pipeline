import os
import sys


def pytest_configure(config):
    # Ensure the repository root (one level up from tests/) is on sys.path so
    # imports like `import src.rag_pipeline` work consistently under pytest.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)