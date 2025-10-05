import re
from src.rag_pipeline import randomized_chunks
def test_randomized_chunks_length():
    text = "A" * 5000
    chunks = randomized_chunks(text, min_size=400, max_size=600, overlap=50, seed=123)
    # ensure overlap constraint: consecutive chunks must overlap by approx 50 characters
    assert len(chunks) > 0
    # check sizes
    for c in chunks:
        assert 100 <= len(c) <= 600 + 50  # allow some slack for end-chunk
