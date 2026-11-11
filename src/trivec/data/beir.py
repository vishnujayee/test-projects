"""Download and parse BEIR datasets from the original BEIR mirror (one zip per dataset).

Zip layout (same for every BEIR dataset):
    <name>/corpus.jsonl         {"_id", "title", "text", ...}
    <name>/queries.jsonl        {"_id", "text", ...}
    <name>/qrels/test.tsv       query-id  corpus-id  score

The dataset "version" we record is the SHA-256 of the zip file. Only queries that have
relevance labels in the test qrels are kept; that is what nDCG is computed on.
"""

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

Qrels = dict[str, dict[str, int]]
"""query id -> {doc id -> relevance grade}"""


@dataclass(frozen=True)
class BeirDataset:
    name: str
    zip_sha256: str
    corpus_ids: list[str]
    corpus_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: Qrels

    @property
    def revision(self) -> str:
        return self.zip_sha256


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(name: str, downloads_dir: Path) -> Path:
    """Fetch <name>.zip once; later calls reuse the file."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    target = downloads_dir / f"{name}.zip"
    if target.exists():
        return target
    url = f"{BEIR_BASE_URL}/{name}.zip"
    tmp = target.with_suffix(".zip.part")
    with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(target)
    return target


def extract_zip(zip_path: Path, raw_dir: Path) -> Path:
    """Extract into raw_dir/<name>/ (its own directory: downloaded data is untrusted)."""
    out = raw_dir / zip_path.stem
    if (out / "corpus.jsonl").exists():
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            # refuse anything that would escape the target directory
            if member.startswith("/") or ".." in Path(member).parts:
                raise ValueError(f"unsafe zip member: {member}")
        z.extractall(raw_dir)
    if not (out / "corpus.jsonl").exists():
        raise FileNotFoundError(f"{zip_path} did not contain {zip_path.stem}/corpus.jsonl")
    return out


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    with path.open(encoding="utf-8") as f:
        header = f.readline()  # "query-id\tcorpus-id\tscore"
        if not header.lower().startswith("query-id"):
            raise ValueError(f"unexpected qrels header in {path}: {header!r}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            qid, did, score = parts
            qrels.setdefault(qid, {})[did] = int(float(score))
    return qrels


def load_beir(name: str, cache_root: Path, split: str = "test") -> BeirDataset:
    zip_path = download_zip(name, cache_root / "downloads")
    folder = extract_zip(zip_path, cache_root / "raw")

    corpus_rows = _read_jsonl(folder / "corpus.jsonl")
    corpus_ids = [str(r["_id"]) for r in corpus_rows]
    corpus_texts = [f"{r.get('title', '')} {r.get('text', '')}".strip() for r in corpus_rows]

    qrels = read_qrels(folder / "qrels" / f"{split}.tsv")
    query_rows = _read_jsonl(folder / "queries.jsonl")
    kept = [(str(r["_id"]), str(r["text"])) for r in query_rows if str(r["_id"]) in qrels]
    query_ids = [q for q, _ in kept]
    query_texts = [t for _, t in kept]

    return BeirDataset(
        name=name,
        zip_sha256=sha256_of(zip_path),
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
    )
