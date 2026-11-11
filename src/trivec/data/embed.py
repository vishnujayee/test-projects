"""Turn texts into L2-normalised float32 embeddings with a pinned sentence-transformers model.

Defaults are chosen for a laptop with little free RAM: small batches and a few threads.
Neither changes the embeddings, only how long they take.
"""

import re

import numpy as np

from trivec.quantizers.base import FloatArray

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def resolve_model_revision(model_name: str, revision: str | None) -> str:
    """Return the exact commit sha for `revision` (or the current default branch).

    A full sha is already exact, so it needs no network call.
    """
    if revision and _FULL_SHA.match(revision):
        return revision
    from huggingface_hub import HfApi

    sha = HfApi().model_info(model_name, revision=revision).sha
    if not sha:
        raise RuntimeError(f"could not resolve revision for {model_name}")
    return str(sha)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def embed_texts(
    texts: list[str],
    model_name: str,
    revision: str,
    *,
    batch_size: int = 16,
    num_threads: int = 4,
) -> FloatArray:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(num_threads)
    model = SentenceTransformer(model_name, revision=revision, device="cpu")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)
