from typing import List


def embed_text(model_embed, texts: List[str]) -> List[List[float]]:
    """Enhanced embedding with normalization."""
    # Use a list comprehension to ensure the input is always a list
    embeddings = model_embed.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.tolist()