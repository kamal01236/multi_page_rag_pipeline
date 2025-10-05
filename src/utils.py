\"\"\"Utility helpers (small functions that might be reused)\"\"\"
from typing import List
def chunk_to_docs(chunks: List[str], source: str, title: str):
    docs = []
    for i, c in enumerate(chunks):
        docs.append({'page_content': c, 'metadata': {'source': source, 'title': title, 'chunk_index': i}})
    return docs
