"""Embedding + ChromaDB storage (supports code + documents)."""
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from codesage.parser import build_embedding_text

_MODEL = None
COLLECTION_PREFIX = "codesage"


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _MODEL


def get_client(db_path: str = "data/chroma_db"):
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def _collection_name(user_id: str) -> str:
    return f"{COLLECTION_PREFIX}_{user_id}"


def index_items(items: list[dict], user_id: str, db_path: str = "data/chroma_db") -> int:
    if not items:
        return 0
    client = get_client(db_path)
    try:
        client.delete_collection(_collection_name(user_id))
    except Exception:
        pass
    coll = client.create_collection(_collection_name(user_id))

    model = get_model()
    texts = [build_embedding_text(i) for i in items]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids, metadatas = [], []
    for idx, i in enumerate(items):
        ids.append(f"{i['kind']}::{i['file']}::{i.get('name','chunk')}::{idx}")
        meta = {
            "kind": i["kind"], "type": i["type"],
            "name": i.get("name", "chunk"), "file": i["file"],
            "text": i["text"],
        }
        if i["kind"] == "code":
            meta["line_start"] = i.get("line_start", 0)
            meta["line_end"]   = i.get("line_end", 0)
            meta["docstring"]  = (i.get("docstring") or "")[:500]
        else:
            meta["page"] = i.get("page", 1)
        metadatas.append(meta)

    coll.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(items)


def retrieve(query: str, user_id: str, top_k: int = 6,
             db_path: str = "data/chroma_db",
             max_distance: float = 1.4) -> list[dict]:
    client = get_client(db_path)
    try:
        coll = client.get_collection(_collection_name(user_id))
    except Exception:
        return []

    q_emb = get_model().encode([query]).tolist()
    res = coll.query(query_embeddings=q_emb, n_results=top_k * 2)

    hits = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        dist = res["distances"][0][i] if "distances" in res else None
        if dist is not None and dist > max_distance:
            continue
        hits.append({
            "kind": md.get("kind", "code"),
            "type": md.get("type", "function"),
            "name": md.get("name", ""),
            "file": md.get("file", ""),
            "line_start": md.get("line_start", 0),
            "line_end": md.get("line_end", 0),
            "page": md.get("page", 1),
            "text": md.get("text", ""),
            "docstring": md.get("docstring", ""),
            "score": dist,
        })
        if len(hits) >= top_k:
            break
    return hits


def list_all(user_id: str, db_path: str = "data/chroma_db") -> list[dict]:
    client = get_client(db_path)
    try:
        coll = client.get_collection(_collection_name(user_id))
    except Exception:
        return []
    data = coll.get()
    out = []
    for md in data["metadatas"]:
        if md.get("kind") != "code":
            continue
        out.append({
            "name": md.get("name", ""),
            "file": md.get("file", ""),
            "line_start": md.get("line_start", 0),
            "text": md.get("text", ""),
            "type": md.get("type", "function"),
        })
    return out