#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEC Word-Shift Analyzer
-----------------------

Reads a GEC reference and system outputs (A, B, C), trains separate Word2Vec models,
computes per-word cosine similarities between each system and the reference, and
visualizes semantic shifts (PCA & t-SNE) for A and B.

Usage
-----
python gec_shift_analysis.py \
  --ref path/to/reference.txt \
  --sysA path/to/system_A.txt \
  --sysB path/to/system_B.txt \
  --sysC path/to/system_C.txt \
  --outdir results/

Outputs
-------
- TXT & CSV files with cosine similarities per shared word:
    similarities_A.txt, similarities_B.txt, similarities_C.txt
    similarities_A.csv, similarities_B.csv, similarities_C.csv
- Global summary file: summary.txt
- PCA & t-SNE scatter plots (PNG) showing semantic shifts for A and B:
    shift_pca_A.png, shift_tsne_A.png, shift_pca_B.png, shift_tsne_B.png

Σημείωση στα Ελληνικά
---------------------
Το script οπτικοποιεί τις ενσωματώσεις λέξεων για τα A, B με PCA/t-SNE και
παρουσιάζει μετατοπίσεις σε σχέση με το reference (GEC). Οι μετατοπίσεις
απεικονίζονται ως βέλη από τη θέση του reference προς τη θέση κάθε συστήματος.

Dependencies
------------
- gensim
- numpy
- pandas
- scikit-learn
- matplotlib

All are commonly available in many Python environments. If gensim is not available,
the script will exit with a clear message.
"""
import argparse
import re
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from gensim.models import Word2Vec
except Exception as e:
    sys.stderr.write("ERROR: gensim is required (pip install gensim).\\n")
    raise

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm

# ---------------------------
# Utilities
# ---------------------------

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿΑ-Ωα-ωάέίόήύώϊϋΰΐά-ώ]+(?:'[A-Za-z]+)?", re.UNICODE)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def sent_tokenize(text: str) -> List[str]:
    # Simple sentence split; keeps things robust without external models
    return re.split(r"(?<=[\\.\\!\\?])\\s+", text.strip())

def word_tokenize(sentence: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(sentence)]

def corpus_to_sentences(text: str) -> List[List[str]]:
    return [word_tokenize(s) for s in sent_tokenize(text) if s.strip()]

def train_w2v(sentences: List[List[str]], vector_size=200, window=5, min_count=2, epochs=50, workers=2, seed=42) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # skip-gram generally better for rare words
        workers=workers,
        seed=seed,
        epochs=epochs,
        negative=10,
    )
    return model

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = norm(u)
    nv = norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))

def shared_vocab(base, other) -> List[str]:
    return sorted(list(set(base).intersection(set(other))))

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_similarity_report(sim_df: pd.DataFrame, path_txt: Path, path_csv: Path, label: str):
    # Sort by similarity (ascending) to highlight biggest shifts first
    sim_df_sorted = sim_df.sort_values("cosine_to_ref").reset_index(drop=True)
    with path_txt.open("w", encoding="utf-8") as f:
        f.write(f"# Cosine similarity vs reference (lower = larger shift): {label}\\n")
        for _, row in sim_df_sorted.iterrows():
            f.write(f"{row['word']}\t{row['cosine_to_ref']:.4f}\\n")
    sim_df_sorted.to_csv(path_csv, index=False, encoding="utf-8")

def plot_shifts_2d(
    ref_vecs: np.ndarray, sys_vecs: np.ndarray, words: List[str],
    method: str, outpath: Path, title: str, max_points: int = 300
):
    # Sample to avoid overcrowding
    if len(words) > max_points:
        idx = np.random.RandomState(42).choice(len(words), size=max_points, replace=False)
        ref_vecs = ref_vecs[idx]
        sys_vecs = sys_vecs[idx]
        words = [words[i] for i in idx]

    # Stack and project
    X = np.vstack([ref_vecs, sys_vecs])

    if method.lower() == "pca":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        proj = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    elif method.lower() == "tsne":
        proj = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=min(30, max(5, len(words)//10 or 5))).fit_transform(X)
    else:
        raise ValueError("Unknown method: choose 'pca' or 'tsne'")

    n = len(words)
    ref_2d = proj[:n]
    sys_2d = proj[n:]

    plt.figure(figsize=(10, 10))
    plt.scatter(ref_2d[:, 0], ref_2d[:, 1], alpha=0.6, label="Reference")
    plt.scatter(sys_2d[:, 0], sys_2d[:, 1], alpha=0.6, label="System")
    # Draw arrows to show shifts
    for i in range(n):
        plt.arrow(ref_2d[i, 0], ref_2d[i, 1],
                  sys_2d[i, 0] - ref_2d[i, 0],
                  sys_2d[i, 1] - ref_2d[i, 1],
                  head_width=0.02, head_length=0.03, alpha=0.4, length_includes_head=True)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ---------------------------
# Main pipeline
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="GEC Word-Shift Analyzer")
    parser.add_argument("--ref", required=True, type=Path, help="Path to GEC reference .txt")
    parser.add_argument("--sysA", required=True, type=Path, help="Path to system A output .txt")
    parser.add_argument("--sysB", required=True, type=Path, help="Path to system B output .txt")
    parser.add_argument("--sysC", required=False, type=Path, help="Path to system C output .txt")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--vector_size", type=int, default=200, help="Word2Vec vector size")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window size")
    parser.add_argument("--min_count", type=int, default=2, help="Word2Vec min_count")
    parser.add_argument("--epochs", type=int, default=50, help="Word2Vec epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_words_viz", type=int, default=300, help="Max words to visualize per plot")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # Read corpora
    ref_text = read_text(args.ref)
    A_text = read_text(args.sysA)
    B_text = read_text(args.sysB)
    C_text = read_text(args.sysC) if args.sysC else ""

    # Tokenize to sentences
    ref_sents = corpus_to_sentences(ref_text)
    A_sents = corpus_to_sentences(A_text)
    B_sents = corpus_to_sentences(B_text)
    C_sents = corpus_to_sentences(C_text) if C_text else []

    # Train one model per corpus
    print("Training Word2Vec models...", file=sys.stderr)
    ref_model = train_w2v(ref_sents, vector_size=args.vector_size, window=args.window, min_count=args.min_count, epochs=args.epochs, seed=args.seed)
    A_model   = train_w2v(A_sents,   vector_size=args.vector_size, window=args.window, min_count=args.min_count, epochs=args.epochs, seed=args.seed)
    B_model   = train_w2v(B_sents,   vector_size=args.vector_size, window=args.window, min_count=args.min_count, epochs=args.epochs, seed=args.seed)
    C_model   = train_w2v(C_sents,   vector_size=args.vector_size, window=args.window, min_count=args.min_count, epochs=args.epochs, seed=args.seed) if C_sents else None

    # Shared vocabularies
    def compute_similarities(sys_model, label: str) -> pd.DataFrame:
        common = shared_vocab(ref_model.wv.index_to_key, sys_model.wv.index_to_key)
        data = []
        for w in common:
            sim = cosine(ref_model.wv[w], sys_model.wv[w])
            data.append((w, sim))
        df = pd.DataFrame(data, columns=["word", "cosine_to_ref"])
        df["system"] = label
        return df

    sim_A = compute_similarities(A_model, "A")
    sim_B = compute_similarities(B_model, "B")
    sim_frames = [sim_A, sim_B]

    if C_model is not None:
        sim_C = compute_similarities(C_model, "C")
        sim_frames.append(sim_C)
    else:
        sim_C = pd.DataFrame(columns=["word", "cosine_to_ref", "system"])

    # Save reports
    save_similarity_report(sim_A, args.outdir / "similarities_A.txt", args.outdir / "similarities_A.csv", "A")
    save_similarity_report(sim_B, args.outdir / "similarities_B.txt", args.outdir / "similarities_B.csv", "B")
    if C_model is not None and not sim_C.empty:
        save_similarity_report(sim_C, args.outdir / "similarities_C.txt", args.outdir / "similarities_C.csv", "C")

    # Summary file
    summary_lines = []
    for label, df in [("A", sim_A), ("B", sim_B), ("C", sim_C)]:
        if df.empty:
            continue
        summary_lines.append(f"System {label}: {len(df)} shared words with reference.")
        summary_lines.append(f"  Mean cosine: {df['cosine_to_ref'].mean():.4f}")
        summary_lines.append(f"  Median cosine: {df['cosine_to_ref'].median():.4f}")
        summary_lines.append(f"  10 lowest-similarity words: {', '.join(df.nsmallest(10, 'cosine_to_ref')['word'].tolist())}")
        summary_lines.append("")
    (args.outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # Prepare vectors for visualization (A and B only, as requested)
    def prepare_vectors_for_viz(sys_model, label: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        common = shared_vocab(ref_model.wv.index_to_key, sys_model.wv.index_to_key)
        # Optionally focus on the most shifted words to make plots more informative
        # We'll pick top-N lowest similarities
        df = pd.DataFrame({
            "word": common,
            "cosine": [cosine(ref_model.wv[w], sys_model.wv[w]) for w in common]
        }).sort_values("cosine").reset_index(drop=True)

        max_n = min(len(df), args.max_words_viz)
        chosen = df.head(max_n)["word"].tolist()  # most shifted
        ref_vecs = np.stack([ref_model.wv[w] for w in chosen])
        sys_vecs = np.stack([sys_model.wv[w] for w in chosen])
        return ref_vecs, sys_vecs, chosen

    # A
    refA, sysA, wordsA = prepare_vectors_for_viz(A_model, "A")
    plot_shifts_2d(refA, sysA, wordsA, method="pca",  outpath=args.outdir / "shift_pca_A.png",  title="Semantic Shifts vs Reference (A) - PCA", max_points=args.max_words_viz)
    plot_shifts_2d(refA, sysA, wordsA, method="tsne", outpath=args.outdir / "shift_tsne_A.png", title="Semantic Shifts vs Reference (A) - t-SNE", max_points=args.max_words_viz)

    # B
    refB, sysB, wordsB = prepare_vectors_for_viz(B_model, "B")
    plot_shifts_2d(refB, sysB, wordsB, method="pca",  outpath=args.outdir / "shift_pca_B.png",  title="Semantic Shifts vs Reference (B) - PCA", max_points=args.max_words_viz)
    plot_shifts_2d(refB, sysB, wordsB, method="tsne", outpath=args.outdir / "shift_tsne_B.png", title="Semantic Shifts vs Reference (B) - t-SNE", max_points=args.max_words_viz)

    print("Done. Results written to", args.outdir, file=sys.stderr)

if __name__ == "__main__":
    main()
