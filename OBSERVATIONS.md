# Observations

After running the pipeline (demo), please record the following observations:

- Did the LLM produce a more readable, concise answer compared to raw chunk concatenation?
- Did it correctly combine information from multiple chunks or pages when relevant?
- Were sources correctly cited in the LLM output?

Typical expected results:
- RetrievalQA will yield a concise answer and list sources when LangChain + LLM are available.
- The TF-IDF fallback will return raw chunks; answers will be less synthesized and will need manual reading.

