"""The Librarian: topic-restricted paper search -> short notes -> searchable with OUR index.

Sources: arXiv API (Atom XML) and Semantic Scholar Graph API. Both free, no key.
Notes live in lab/library/notes/<id>.md. They are embedded and indexed with TriVec's own
ternary PackedPopcountIndex (dogfooding), so the Proposer can search them.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

from trivec.index.packed import PackedPopcountIndex
from trivec.lab.state import atomic_write_text
from trivec.quantizers.base import FloatArray
from trivec.quantizers.ternary import TernaryQuantizer

ARXIV_URL = "https://export.arxiv.org/api/query"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
TOPIC_WORDS = (
    "quantiz",
    "compress",
    "binary",
    "ternary",
    "hashing",
    "nearest neighbor",
    "nearest neighbour",
    "similarity search",
    "vector",
    "embedding",
    "retrieval",
)


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    year: int | None
    url: str
    source: str  # "arxiv" | "semanticscholar"


Fetcher = Callable[[str, int], list[Paper]]
Embedder = Callable[[Sequence[str]], FloatArray]


def on_topic(paper: Paper) -> bool:
    text = f"{paper.title} {paper.abstract}".lower()
    return sum(w in text for w in TOPIC_WORDS) >= 2


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def fetch_arxiv(query: str, max_results: int, client: httpx.Client | None = None) -> list[Paper]:
    params: dict[str, str | int] = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    c = client or httpx.Client(timeout=30.0)
    r = c.get(ARXIV_URL, params=params)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", ns):
        ident = (entry.findtext("a:id", default="", namespaces=ns) or "").rsplit("/", 1)[-1]
        title = " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split())
        abstract = " ".join((entry.findtext("a:summary", default="", namespaces=ns) or "").split())
        published = entry.findtext("a:published", default="", namespaces=ns) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        papers.append(
            Paper(
                f"arxiv_{_slug(ident)}",
                title,
                abstract,
                year,
                f"https://arxiv.org/abs/{ident}",
                "arxiv",
            )
        )
    return papers


def fetch_semantic_scholar(
    query: str, max_results: int, client: httpx.Client | None = None
) -> list[Paper]:
    params: dict[str, str | int] = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,url,externalIds",
    }
    c = client or httpx.Client(timeout=30.0)
    r = c.get(S2_URL, params=params)
    r.raise_for_status()
    papers: list[Paper] = []
    for item in r.json().get("data", []):
        pid = str(item.get("paperId", ""))
        papers.append(
            Paper(
                paper_id=f"s2_{pid[:16]}",
                title=str(item.get("title") or ""),
                abstract=str(item.get("abstract") or ""),
                year=item.get("year"),
                url=str(item.get("url") or ""),
                source="semanticscholar",
            )
        )
    return papers


def note_text(paper: Paper, summary: str | None = None) -> str:
    body = summary or f"**Abstract.** {paper.abstract}"
    return (
        f"# {paper.title}\n\n"
        f"- id: {paper.paper_id}\n- year: {paper.year}\n- source: {paper.source}\n- link: {paper.url}\n\n"
        f"{body}\n"
    )


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.notes_dir = root / "notes"
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def note_path(self, paper_id: str) -> Path:
        return self.notes_dir / f"{paper_id}.md"

    def add_papers(
        self, papers: Sequence[Paper], summarize: Callable[[Paper], str] | None = None
    ) -> int:
        added = 0
        for p in papers:
            if not on_topic(p) or self.note_path(p.paper_id).exists() or not p.abstract:
                continue
            summary = summarize(p) if summarize else None
            atomic_write_text(self.note_path(p.paper_id), note_text(p, summary))
            atomic_write_text(self.notes_dir / f"{p.paper_id}.json", json.dumps(asdict(p)))
            added += 1
        return added

    def collect(
        self, queries: Sequence[str], max_per_query: int, fetchers: Sequence[Fetcher]
    ) -> int:
        papers: list[Paper] = []
        for q in queries:
            for fetch in fetchers:
                try:
                    papers.extend(fetch(q, max_per_query))
                except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
                    (self.root / "librarian_errors.log").open("a", encoding="utf-8").write(
                        f"{q}: {exc}\n"
                    )
        return self.add_papers(papers)

    def notes(self) -> list[tuple[str, str]]:
        return [
            (p.stem, p.read_text(encoding="utf-8")) for p in sorted(self.notes_dir.glob("*.md"))
        ]

    def search(self, query: str, embed: Embedder, k: int = 5) -> list[tuple[str, str]]:
        """Search notes with TriVec's own ternary index (symmetric popcount)."""
        notes = self.notes()
        if not notes:
            return []
        # embed title + body only; the "- id/year/source/link" lines are shared boilerplate
        texts = [
            "\n".join(ln for ln in t.splitlines() if not ln.startswith("- ")) for _, t in notes
        ]
        corpus = embed(texts)
        q = TernaryQuantizer(alpha=0.7)
        q.fit(corpus, seed=0)
        index = PackedPopcountIndex(q, q.encode(corpus))
        ids, _ = index.search(embed([query]), k=min(k, len(notes)))
        return [notes[int(i)] for i in ids[0]]


def hash_embedder(dim: int = 1024) -> Embedder:
    """Deterministic bag-of-words embedder for tests and offline use (no model download)."""

    def embed(texts: Sequence[str]) -> FloatArray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                digest = hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                out[i, h % dim] += 1.0 if h >> 63 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(out / norms, dtype=np.float32)

    return embed


def model_embedder(model_name: str, revision: str | None = None) -> Embedder:
    from trivec.data.embed import embed_texts, resolve_model_revision

    rev = revision or resolve_model_revision(model_name, None)

    def embed(texts: Sequence[str]) -> FloatArray:
        return embed_texts(list(texts), model_name, rev)

    return embed
