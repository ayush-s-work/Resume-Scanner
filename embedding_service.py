from sentence_transformers import SentenceTransformer
from typing import List


MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text.
    Returns a list of floats.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


