def split_text(text: str, max_length=500, overlap=50):
    """
    Делит текст на куски примерно по max_length символов
    с overlap для контекста между кусками.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap
    return chunks
