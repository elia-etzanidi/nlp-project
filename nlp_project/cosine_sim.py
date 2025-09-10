"""
Phase 2 scoring: embedding-based similarity + PCA/t-SNE visualization.

Reads files in a folder (default: ./outputs) with the naming:
  textN_orig.txt      (REQUIRED)  -> original text
  textN_pipeA.txt     (REQUIRED)  -> pipeline A output
  textN_pipeB.txt     (REQUIRED)  -> pipeline B output
  textN_pipeC.txt     (REQUIRED)  -> pipeline C output
  textN_gec.txt       (OPTIONAL)  -> GEC reference (for extra context/plotting)

Computes:
  - SentenceTransformer embeddings (all-MiniLM-L6-v2, 384-d)
  - Cosine similarity: cos(orig, pipeA/B/C) per textN
Saves:
  - JSON with scores (if --outjson provided)
  - PCA plot (embeddings_pca.png)
  - t-SNE plot (embeddings_tsne.png)

Click-to-run default: reads ./outputs and writes plots into ./outputs.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re
import json
import math
import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# You’ll need: pip install sentence-transformers scikit-learn matplotlib
from sentence_transformers import SentenceTransformer, util as st_util

# ---------- file IO ----------
RE_PAT = re.compile(r"^(text\d+?)_(orig|pipeA|pipeB|pipeC|gec)\.txt$")

def scan_folder(indir: str | Path) -> Dict[str, Dict[str, str]]:
    p = Path(indir)
    found: Dict[str, Dict[str, str]] = {}
    for f in p.glob("*.txt"):
        m = RE_PAT.match(f.name)
        if not m:
            continue
        tid, kind = m.group(1), m.group(2)
        found.setdefault(tid, {})
        found[tid][kind] = f.read_text(encoding="utf-8").strip()
    return found

# ---------- embeddings + cosine ----------
@dataclass
class EmbeddingResult:
    text_id: str
    vectors: Dict[str, np.ndarray]      # keys: 'orig', 'pipeA', 'pipeB', 'pipeC', optional 'gec'
    cos_sim: Dict[str, float]           # keys: 'pipeA','pipeB','pipeC' vs 'orig'

def embed_texts(model: SentenceTransformer, blobs: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Embed each present string value into a vector (L2 normalized).
    """
    keys = [k for k in ["orig","pipeA","pipeB","pipeC","gec"] if k in blobs]
    texts = [blobs[k] for k in keys]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return {k: emb for k, emb in zip(keys, embs)}

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are normalized, so cosine = dot
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def score_folder(indir: str | Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[List[EmbeddingResult], Dict]:
    files = scan_folder(indir)
    model = SentenceTransformer(model_name)

    results: List[EmbeddingResult] = []
    summary: Dict[str, Dict[str, float]] = {}  # per text_id similarities

    for tid, parts in sorted(files.items()):
        # require original + all three pipes to produce a full row
        missing = [k for k in ["orig","pipeA","pipeB","pipeC"] if k not in parts]
        if missing:
            summary[tid] = {"error": f"Missing required files: {', '.join(missing)}"}
            continue

        vecs = embed_texts(model, parts)
        cs = {
            "pipeA": cosine(vecs["orig"], vecs["pipeA"]),
            "pipeB": cosine(vecs["orig"], vecs["pipeB"]),
            "pipeC": cosine(vecs["orig"], vecs["pipeC"]),
        }

        results.append(EmbeddingResult(text_id=tid, vectors=vecs, cos_sim=cs))
        summary[tid] = cs

    return results, summary

# ---------- visualization ----------
def _collect_points_for_plot(items: List[EmbeddingResult]) -> Tuple[np.ndarray, List[str]]:
    """
    Build a matrix of all points and labels for plotting.
    Labels like: "text1_orig", "text1_pipeA", ...
    """
    rows: List[np.ndarray] = []
    labels: List[str] = []
    for r in items:
        for key in ["orig","gec","pipeA","pipeB","pipeC"]:
            if key in r.vectors:
                rows.append(r.vectors[key])
                labels.append(f"{r.text_id}_{key}")
    X = np.vstack(rows) if rows else np.zeros((0, 384), dtype=np.float32)
    return X, labels

def plot_pca(X: np.ndarray, labels: List[str], outpath: Path):
    if X.shape[0] == 0:
        return
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1])
    for (x, y, lab) in zip(X2[:,0], X2[:,1], labels):
        plt.annotate(lab, (x, y), fontsize=8, xytext=(2, 2), textcoords="offset points")
    plt.title("PCA of Text Embeddings")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_tsne(X: np.ndarray, labels: List[str], outpath: Path, perplexity: float = 5.0):
    if X.shape[0] == 0:
        return
    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(2, X.shape[0]-1)), init="pca", random_state=42)
    X2 = tsne.fit_transform(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1])
    for (x, y, lab) in zip(X2[:,0], X2[:,1], labels):
        plt.annotate(lab, (x, y), fontsize=8, xytext=(2, 2), textcoords="offset points")
    plt.title("t-SNE of Text Embeddings")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Embedding similarity + PCA/t-SNE visualization")
    parser.add_argument("--indir", type=str, default="outputs", help="Folder with textN_orig.txt, textN_pipeA/B/C.txt, optional textN_gec.txt")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    parser.add_argument("--outjson", type=str, default="", help="Optional: path to write similarities JSON")
    parser.add_argument("--no_plots", action="store_true", help="Disable PCA/t-SNE plot generation")
    args = parser.parse_args()

    outdir = Path(args.indir)

    results, summary = score_folder(outdir, model_name=args.model)

    # pretty print
    for tid in sorted(summary.keys()):
        row = summary[tid]
        if "error" in row:
            print(f"{tid}: {row['error']}")
            continue
        print(f"\n{tid}")
        for k in ["pipeA","pipeB","pipeC"]:
            if k in row:
                print(f"  cosine(orig, {k}): {row[k]:.4f}")

    if args.outjson:
        Path(args.outjson).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON to: {Path(args.outjson).resolve()}")

    if not args.no_plots:
        X, labels = _collect_points_for_plot(results)
        if X.shape[0] == 0:
            print("\nNo points to plot.")
        else:
            plot_pca(X, labels, outdir / "embeddings_pca.png")
            plot_tsne(X, labels, outdir / "embeddings_tsne.png")
            print(f"\nSaved plots to:\n  { (outdir / 'embeddings_pca.png').resolve() }\n  { (outdir / 'embeddings_tsne.png').resolve() }")