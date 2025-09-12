"""
Simple embedding similarity + PCA/t-SNE.

Originals:
  texts/text1.txt, texts/text2.txt, ...

Pipelines:
  outputs/text1_pipeA.txt, outputs/text1_pipeB.txt, outputs/text1_pipeC.txt, ...
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ---------- config (relative to this file) ----------
SCRIPT_DIR  = Path(__file__).resolve().parent
ORIG_DIR    = SCRIPT_DIR / "texts"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
WRITE_JSON  = True
MAKE_PLOTS  = True
TSNE_PERPLEXITY = 5.0

# ---------- tiny data loader ----------
def load_data(orig_dir: Path, out_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns:
      {
        'text1': {'orig': '...', 'pipeA': '...', 'pipeB': '...', 'pipeC': '...'}, # pipes present only if files exist
        'text2': {...},
      }
    """
    data: Dict[str, Dict[str, str]] = {}

    # 1) Read any pipeline files present
    for f in out_dir.glob("text*_pipe*.txt"):
        stem = f.stem  # e.g., "text1_pipeA"
        try:
            tid, kind = stem.split("_", 1)  # ["text1", "pipeA"]
        except ValueError:
            # Skip unexpected names
            continue
        d = data.setdefault(tid, {})
        d[kind] = f.read_text(encoding="utf-8").strip()

    # 2) Read originals for any text id that had at least one pipeline file
    for tid in list(data.keys()):
        opath = orig_dir / f"{tid}.txt"  # <-- ORIGINAL READ HERE
        if opath.exists():
            data[tid]["orig"] = opath.read_text(encoding="utf-8").strip()
        else:
            data[tid]["__error__"] = f"Missing original: {opath}"

    return data

# ---------- embeddings + cosine ----------
@dataclass
class EmbeddingResult:
    text_id: str
    vectors: Dict[str, np.ndarray]      # keys like 'orig', 'pipeA', 'pipeB', 'pipeC'
    cos_sim: Dict[str, float]           # cosine(orig, pipeX)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings normalized, so cosine == dot
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def score(data: Dict[str, Dict[str, str]], model_name: str) -> Tuple[List[EmbeddingResult], Dict[str, Dict[str, float]]]:
    model = SentenceTransformer(model_name)
    results: List[EmbeddingResult] = []
    summary: Dict[str, Dict[str, float]] = {}

    for tid, parts in sorted(data.items()):
        if "__error__" in parts:
            summary[tid] = {"error": parts["__error__"]}
            continue
        if "orig" not in parts:
            summary[tid] = {"error": "Missing original"}
            continue

        # Embed only what's present
        keys = ["orig"] + [k for k in ["pipeA","pipeB","pipeC"] if k in parts]
        texts = [parts[k] for k in keys]
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        vecs = {k: v for k, v in zip(keys, embs)}

        # Cosine vs original
        cs = {k: cosine(vecs["orig"], vecs[k]) for k in keys if k != "orig"}
        results.append(EmbeddingResult(text_id=tid, vectors=vecs, cos_sim=cs))
        summary[tid] = cs

    return results, summary

# ---------- plotting ----------
def _collect_points_for_plot(items: List[EmbeddingResult]):
    rows: List[np.ndarray] = []
    labels: List[str] = []
    for r in items:
        for key in ["orig","pipeA","pipeB","pipeC"]:
            if key in r.vectors:
                rows.append(r.vectors[key])
                labels.append(f"{r.text_id}_{key}")
    X = np.vstack(rows) if rows else np.zeros((0, 384), dtype=np.float32)
    return X, labels

def plot_pca(X: np.ndarray, labels: List[str], outpath: Path):
    if X.shape[0] == 0: return
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1])
    for (x, y, lab) in zip(X2[:,0], X2[:,1], labels):
        plt.annotate(lab, (x, y), fontsize=8, xytext=(2, 2), textcoords="offset points")
    plt.title("PCA of Text Embeddings")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_tsne(X: np.ndarray, labels: List[str], outpath: Path, perplexity: float = 5.0):
    if X.shape[0] == 0: return
    p = min(perplexity, max(2, X.shape[0]-1))
    X2 = TSNE(n_components=2, perplexity=p, init="pca", random_state=42).fit_transform(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1])
    for (x, y, lab) in zip(X2[:,0], X2[:,1], labels):
        plt.annotate(lab, (x, y), fontsize=8, xytext=(2, 2), textcoords="offset points")
    plt.title("t-SNE of Text Embeddings")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------- run from VS Code ----------
def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Originals: {ORIG_DIR}")
    print(f"Pipelines: {OUTPUTS_DIR}")

    data = load_data(ORIG_DIR, OUTPUTS_DIR)

    # Quick inventory so you see what was read
    print("\n[Inventory]")
    for tid, parts in sorted(data.items()):
        present = [k for k in parts.keys() if not k.startswith("__")]
        note = parts.get("__error__", "")
        print(f"  {tid}: {sorted(present)}" + (f" | {note}" if note else ""))

    results, summary = score(data, MODEL_NAME)

    print("\n[Cosine similarities]")
    for tid in sorted(summary.keys()):
        row = summary[tid]
        if "error" in row:
            print(f"  {tid}: {row['error']}")
            continue
        print(f"  {tid}:")
        for k in ["pipeA","pipeB","pipeC"]:
            if k in row:
                print(f"    cosine(orig, {k}) = {row[k]:.4f}")

    if WRITE_JSON:
        (OUTPUTS_DIR / "similarities.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON -> { (OUTPUTS_DIR / 'similarities.json').resolve() }")

    if MAKE_PLOTS:
        X, labels = _collect_points_for_plot(results)
        if X.shape[0] == 0:
            print("\nNo points to plot.")
        else:
            plot_pca(X, labels, OUTPUTS_DIR / "embeddings_pca.png")
            plot_tsne(X, labels, OUTPUTS_DIR / "embeddings_tsne.png", perplexity=TSNE_PERPLEXITY)
            print(f"Saved plots ->\n  { (OUTPUTS_DIR / 'embeddings_pca.png').resolve() }\n  { (OUTPUTS_DIR / 'embeddings_tsne.png').resolve() }")

if __name__ == "__main__":
    # Run directly in VS Code (Run Python File / F5)
    main()
