import re, numpy as np


def normalize(morph, text):
    return re.sub(r"[^\w\s]", "", text.lower())


def cosine(embedder, text, pts):
    emb = embedder.encode([text])[0]

    sims = np.dot(pts, emb) / (
        np.linalg.norm(pts, axis=1) * np.linalg.norm(emb)
    )

    idx = np.argmax(sims)
    score = sims[idx]

    return idx, score