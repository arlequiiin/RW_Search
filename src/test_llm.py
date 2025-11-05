from src.chunker import split_text
print(len(split_text("a" * 2000, 500, 50)))
