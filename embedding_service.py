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


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple text chunks.
    """

    if not texts:
        raise ValueError("Text list cannot be empty")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()

sections = {
    "summary": "Backend developer with Python and FastAPI experience.",
    "skills": "Python, FastAPI, PostgreSQL, Docker",
    "education": "B.Tech in Computer Science"
}

texts = list(sections.values())
embeddings = generate_embeddings(texts)

for section_name, embedding in zip(sections.keys(), embeddings):
    print(section_name, len(embedding))