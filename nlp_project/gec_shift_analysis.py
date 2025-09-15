"""
GEC Word-Shift Analyzer
-----------------------

Reads a GEC reference and three system outputs (A, B, C), trains separate Word2Vec models,
computes per-word cosine similarities between each system and the reference, and
visualizes semantic shifts (PCA & t-SNE).

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
- CSV files with cosine similarities per shared word:
    similarities_A.csv, similarities_B.csv, similarities_C.csv
- Global summary file: summary.txt
- PCA & t-SNE scatter plots (PNG) showing semantic shifts:
    shift_pca_A.png, shift_tsne_A.png, shift_pca_B.png, shift_tsne_B.png,
    shift_pca_C.png, shift_tsne_C.png

Dependencies
------------
- gensim, numpy, pandas, scikit-learn, matplotlib
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from gensim.models import Word2Vec
except ImportError:
    sys.stderr.write("ERROR: gensim is required (pip install gensim).\n")
    raise

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# Token extraction regex
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿΑ-Ωα-ωάέίόήύώϊϋΰΐά-ώ]+(?:'[A-Za-z]+)?", re.UNICODE)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def sent_tokenize(text: str) -> List[str]:
    return re.split(r"(?<=[\.\!\?])\s+", text.strip())

def word_tokenize(sentence: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(sentence)]

def corpus_to_sentences(text: str) -> List[List[str]]:
    return [word_tokenize(s) for s in sent_tokenize(text) if s.strip()]

def train_w2v(sentences: List[List[str]], vector_size=200, window=5, min_count=2, epochs=50, seed=42) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=2,
        seed=seed,
        epochs=epochs,
        negative=10,
    )

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = norm(u), norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))

def shared_vocab(vocab1, vocab2) -> List[str]:
    return sorted(list(set(vocab1).intersection(set(vocab2))))

def compute_similarities(ref_model: Word2Vec, sys_model: Word2Vec, label: str) -> pd.DataFrame:
    common = shared_vocab(ref_model.wv.index_to_key, sys_model.wv.index_to_key)
    data = [(w, cosine(ref_model.wv[w], sys_model.wv[w])) for w in common]
    df = pd.DataFrame(data, columns=["word", "cosine_to_ref"])
    df["system"] = label
    return df

def print_similarity_report(sim_df: pd.DataFrame, label: str):
    sim_df_sorted = sim_df.sort_values("cosine_to_ref").reset_index(drop=True)
    print(f"\n=== Cosine similarity vs reference ({label}) ===")
    for _, row in sim_df_sorted.iterrows():
        print(f"{row['word']}\t{row['cosine_to_ref']:.4f}")

def plot_shifts_2d(ref_vecs: np.ndarray, sys_vecs: np.ndarray, words: List[str],
                   method: str, outpath: Path, title: str, max_points: int = 300):
    # Sample to avoid overcrowding
    if len(words) > max_points:
        idx = np.random.RandomState(42).choice(len(words), size=max_points, replace=False)
        ref_vecs = ref_vecs[idx]
        sys_vecs = sys_vecs[idx]
        words = [words[i] for i in idx]

    X = np.vstack([ref_vecs, sys_vecs])

    if method.lower() == "pca":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        proj = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    elif method.lower() == "tsne":
        perplexity = min(30, max(5, len(words) // 10 or 5))
        proj = TSNE(n_components=2, random_state=42, init="pca", 
                   learning_rate="auto", perplexity=perplexity).fit_transform(X)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    n = len(words)
    ref_2d, sys_2d = proj[:n], proj[n:]

    plt.figure(figsize=(10, 10))
    plt.scatter(ref_2d[:, 0], ref_2d[:, 1], alpha=0.6, label="Reference")
    plt.scatter(sys_2d[:, 0], sys_2d[:, 1], alpha=0.6, label="System")
    
    # Draw shift arrows
    for i in range(n):
        plt.arrow(ref_2d[i, 0], ref_2d[i, 1],
                  sys_2d[i, 0] - ref_2d[i, 0], sys_2d[i, 1] - ref_2d[i, 1],
                  head_width=0.02, head_length=0.03, alpha=0.4, length_includes_head=True)
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def prepare_vectors_for_viz(ref_model: Word2Vec, sys_model: Word2Vec, 
                           max_words: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    common = shared_vocab(ref_model.wv.index_to_key, sys_model.wv.index_to_key)
    
    # Focus on most shifted words for more informative plots
    df = pd.DataFrame({
        "word": common,
        "cosine": [cosine(ref_model.wv[w], sys_model.wv[w]) for w in common]
    }).sort_values("cosine").head(min(len(common), max_words))
    
    chosen = df["word"].tolist()
    ref_vecs = np.stack([ref_model.wv[w] for w in chosen])
    sys_vecs = np.stack([sys_model.wv[w] for w in chosen])
    return ref_vecs, sys_vecs, chosen

def main():
    parser = argparse.ArgumentParser(description="GEC Word-Shift Analyzer")
    parser.add_argument("--ref", required=True, type=Path, help="Path to GEC reference .txt")
    parser.add_argument("--sysA", required=True, type=Path, help="Path to system A output .txt")
    parser.add_argument("--sysB", required=True, type=Path, help="Path to system B output .txt")
    parser.add_argument("--sysC", required=True, type=Path, help="Path to system C output .txt")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--vector_size", type=int, default=200, help="Word2Vec vector size")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window size")
    parser.add_argument("--min_count", type=int, default=2, help="Word2Vec min_count")
    parser.add_argument("--epochs", type=int, default=50, help="Word2Vec epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_words_viz", type=int, default=300, help="Max words to visualize per plot")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Read and tokenize texts
    ref_sents = corpus_to_sentences(read_text(args.ref))
    A_sents = corpus_to_sentences(read_text(args.sysA))
    B_sents = corpus_to_sentences(read_text(args.sysB))
    C_sents = corpus_to_sentences(read_text(args.sysC))

    # Train Word2Vec models
    print("Training Word2Vec models...", file=sys.stderr)
    w2v_params = {
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "seed": args.seed
    }
    
    ref_model = train_w2v(ref_sents, **w2v_params)
    A_model = train_w2v(A_sents, **w2v_params)
    B_model = train_w2v(B_sents, **w2v_params)
    C_model = train_w2v(C_sents, **w2v_params)

    # Compute similarities
    sim_A = compute_similarities(ref_model, A_model, "A")
    sim_B = compute_similarities(ref_model, B_model, "B")
    sim_C = compute_similarities(ref_model, C_model, "C")

    # Print reports
    for df, label in [(sim_A, "A"), (sim_B, "B"), (sim_C, "C")]:
        print_similarity_report(df, label)

    # Summary statistics
    print("\n=== Summary statistics ===")
    for label, df in [("A", sim_A), ("B", sim_B), ("C", sim_C)]:
        print(f"System {label}: {len(df)} shared words with reference.")
        print(f"  Mean cosine: {df['cosine_to_ref'].mean():.4f}")
        print(f"  Median cosine: {df['cosine_to_ref'].median():.4f}")
        print("  10 lowest-similarity words:",
              ", ".join(df.nsmallest(10, 'cosine_to_ref')['word'].tolist()),"\n")

    # Save CSV files
    sim_A.to_csv(args.outdir / "similarities_A.csv", index=False)
    sim_B.to_csv(args.outdir / "similarities_B.csv", index=False)
    sim_C.to_csv(args.outdir / "similarities_C.csv", index=False)

    # Generate visualizations for all systems
    systems = [
        (A_model, "A", "shift_pca_A.png", "shift_tsne_A.png"),
        (B_model, "B", "shift_pca_B.png", "shift_tsne_B.png"),
        (C_model, "C", "shift_pca_C.png", "shift_tsne_C.png")
    ]
    
    for model, label, pca_file, tsne_file in systems:
        ref_vecs, sys_vecs, words = prepare_vectors_for_viz(ref_model, model, args.max_words_viz)
        
        plot_shifts_2d(ref_vecs, sys_vecs, words, method="pca",
                      outpath=args.outdir / pca_file,
                      title=f"Semantic Shifts vs Reference ({label}) - PCA",
                      max_points=args.max_words_viz)
        
        plot_shifts_2d(ref_vecs, sys_vecs, words, method="tsne",
                      outpath=args.outdir / tsne_file,
                      title=f"Semantic Shifts vs Reference ({label}) - t-SNE",
                      max_points=args.max_words_viz)

    print("Analysis complete. Results written to", args.outdir, file=sys.stderr)

if __name__ == "__main__":
    main()