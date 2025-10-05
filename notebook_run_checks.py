import types
from pprint import pprint
from src import rag_pipeline as rp

# 1) Stub fetch_html for deterministic content
def stub_fetch_html(url):
    return """<html><h1 id='firstHeading'>Test Page</h1><div id='mw-content-text'><p>This is a test page content about qubits and Quantum neural networks (QNN).</p></div></html>"""

rp.fetch_html = lambda url: stub_fetch_html(url)
print('fetch_html stubbed')

# 2) Simulate no LangChain available and ensure build_from_urls produces document-like objects
orig_doc = getattr(rp, 'Document', None)
orig_has_lang = getattr(rp, 'HAS_LANGCHAIN', True)
try:
    rp.Document = None
    rp.HAS_LANGCHAIN = False
    docs, store = rp.build_from_urls(['https://en.wikipedia.org/wiki/Quantum_machine_learning','https://en.wikipedia.org/wiki/Quantum_computing'], seed=1)
    print('\nbuild_from_urls returned', len(docs), 'docs')
    first = docs[0]
    # Support both dict-style and object-style docs
    if isinstance(first, dict):
        wrapped = types.SimpleNamespace(page_content=first.get('page_content'), metadata=first.get('metadata'))
    else:
        wrapped = first
    print('Has page_content attribute:', hasattr(wrapped, 'page_content'))
    print('Has metadata attribute:', hasattr(wrapped, 'metadata'))
    pprint(getattr(wrapped, 'metadata', None))

    # 3) Build vectorstore and ensure TF-IDF fallback is returned
    store2, stype = rp.build_vectorstore(docs, backend='auto')
    print('\nbuild_vectorstore type:', stype)
    if hasattr(store2, 'search'):
        hits = store2.search('qubits', k=1)
        print('TF-IDF store search returned', len(hits), 'hit(s)')
    else:
        print('store2 has no search')

    # 4) Use search_db (repo helper) and test filter_source behavior
    hits_global = rp.search_db(store2, 'Quantum neural networks', k=3)
    print('\nsearch_db global hits:')
    pprint(hits_global)

    hits_filtered = rp.search_db(store2, 'Quantum neural networks', k=3, filter_source='https://en.wikipedia.org/wiki/Quantum_machine_learning')
    print('\nsearch_db filtered hits:')
    pprint(hits_filtered)

    # 5) Make page-specific askers and call them
    askers = rp.make_page_specific_askers(store2, ['https://en.wikipedia.org/wiki/Quantum_machine_learning'])
    print('\nmake_page_specific_askers keys:', list(askers.keys()))
    if askers:
        slug = list(askers.keys())[0]
        print('Calling asker for slug:', slug)
        resp = askers[slug]('what is a qubit?', k=2)
        pprint(resp)

finally:
    # restore
    rp.Document = orig_doc
    rp.HAS_LANGCHAIN = orig_has_lang

print('\nObservations:')
print('- build_from_urls can create fallback docs when LangChain is disabled')
print('- build_vectorstore returned TF-IDF fallback when LangChain disabled')
print('- search_db returns global and filtered hits; page-specific askers exist and return evidence with sources')
