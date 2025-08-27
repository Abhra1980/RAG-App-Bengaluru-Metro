# 3) Chunking
# -----------------------------
def chunk_text(text: str, max_char: int = 500, overlap: int = 100):
    assert max_char > overlap >= 0, "max_char must be > overlap >= 0"
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + max_char]
        chunks.append(piece)
        i += max_char - overlap
    return chunks