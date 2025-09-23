"""
Cosine similarity analyzer
---------------------------

Supports two modes:
1. TXT Mode: Analyzes pipelines A, B, C
2. SEN Mode: Analyzes sentence reconstruction outputs

Usage
-----
cd into nlp_project and run:

TXT Mode:
TEXT 1) python cosine_sim.py txt --ref outputs/text1_gec.txt --sysA outputs/text1_pipeA.txt --sysB outputs/text1_pipeB.txt --sysC outputs/text1_pipeC.txt --outdir results/text1/
TEXT 2) python cosine_sim.py txt --ref outputs/text2_gec.txt --sysA outputs/text2_pipeA.txt --sysB outputs/text2_pipeB.txt --sysC outputs/text2_pipeC.txt --outdir results/text2/

SEN Mode:
python cosine_sim.py sen --sent1_og outputs/sent1_og.txt --sent1_out outputs/sent1_out.txt --sent2_og outputs/sent2_og.txt --sent2_out outputs/sent2_out.txt --outdir results/

Outputs
-------
- PCA & t-SNE scatter plots (PNG) showing semantic shifts

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
    # Adjust min_count for small datasets
    unique_words = len(set(word for sent in sentences for word in sent))
    adjusted_min_count = min(min_count, max(1, unique_words // 10))
    
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=adjusted_min_count,
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
    # Check if we have enough data for visualization
    if len(words) < 3:
        return
    
    # Sample to avoid overcrowding
    if len(words) > max_points:
        idx = np.random.RandomState(42).choice(len(words), size=max_points, replace=False)
        ref_vecs = ref_vecs[idx]
        sys_vecs = sys_vecs[idx]
        words = [words[i] for i in idx]

    X = np.vstack([ref_vecs, sys_vecs])

    if method.lower() == "pca":
        # Check for valid data
        if X.shape[0] < 2 or X.shape[1] < 2:
            return
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        proj = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    elif method.lower() == "tsne":
        # t-SNE needs at least 3 samples and perplexity < n_samples
        if len(words) < 3:
            return
        perplexity = min(5, max(1, (len(words) - 1) // 2))  # Much more conservative
        proj = TSNE(n_components=2, random_state=42, init="pca", 
                   learning_rate="auto", perplexity=perplexity).fit_transform(X)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    n = len(words)
    ref_2d, sys_2d = proj[:n], proj[n:]

    plt.figure(figsize=(10, 10))
    plt.scatter(ref_2d[:, 0], ref_2d[:, 1], alpha=0.6, label="Original" if "Original" in title else "Reference")
    plt.scatter(sys_2d[:, 0], sys_2d[:, 1], alpha=0.6, label="Reconstructed" if "Reconstructed" in title else "System")
    
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
    print(f"\nSaved plot: {outpath}")

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

def run_txt_analysis(args):
    """Run TXT analysis: reference vs systems A, B, C"""
    ref_sents = corpus_to_sentences(read_text(args.ref))
    A_sents = corpus_to_sentences(read_text(args.sysA))
    B_sents = corpus_to_sentences(read_text(args.sysB))
    C_sents = corpus_to_sentences(read_text(args.sysC))

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
        print(f"Pipeline {label}: {len(df)} shared words with reference.")
        print(f"  Mean cosine: {df['cosine_to_ref'].mean():.4f}")
        print(f"  Median cosine: {df['cosine_to_ref'].median():.4f}")

    # Generate visualizations
    systems = [(A_model, "A"), (B_model, "B"), (C_model, "C")]
    for model, label in systems:
        ref_vecs, sys_vecs, words = prepare_vectors_for_viz(ref_model, model, args.max_words_viz)
        
        plot_shifts_2d(ref_vecs, sys_vecs, words, method="pca",
                      outpath=args.outdir / f"shift_pca_{label}.png",
                      title=f"Semantic Shifts vs Reference ({label}) - PCA",
                      max_points=args.max_words_viz)
        
        plot_shifts_2d(ref_vecs, sys_vecs, words, method="tsne",
                      outpath=args.outdir / f"shift_tsne_{label}.png",
                      title=f"Semantic Shifts vs Reference ({label}) - t-SNE",
                      max_points=args.max_words_viz)

def run_sen_analysis(args):
    """Run SEN analysis: original vs reconstructed sentences"""
    sent1_og_sents = corpus_to_sentences(read_text(args.sent1_og))
    sent1_out_sents = corpus_to_sentences(read_text(args.sent1_out))
    sent2_og_sents = corpus_to_sentences(read_text(args.sent2_og))
    sent2_out_sents = corpus_to_sentences(read_text(args.sent2_out))

    w2v_params = {
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "seed": args.seed
    }
    
    sent1_og_model = train_w2v(sent1_og_sents, **w2v_params)
    sent1_out_model = train_w2v(sent1_out_sents, **w2v_params)
    sent2_og_model = train_w2v(sent2_og_sents, **w2v_params)
    sent2_out_model = train_w2v(sent2_out_sents, **w2v_params)

    # Compute similarities (original vs reconstructed)
    sim_sent1 = compute_similarities(sent1_og_model, sent1_out_model, "Sent1")
    sim_sent2 = compute_similarities(sent2_og_model, sent2_out_model, "Sent2")

    # Print reports
    for df, label in [(sim_sent1, "Sent1"), (sim_sent2, "Sent2")]:
        print_similarity_report(df, label)

    # Summary statistics
    print("\n=== Summary statistics ===")
    for label, df in [("Sentence 1", sim_sent1), ("Sentence 2", sim_sent2)]:
        print(f"{label}: {len(df)} shared words between original and reconstructed.")
        if len(df) > 0:
            print(f"  Mean cosine: {df['cosine_to_ref'].mean():.4f}")
            print(f"  Median cosine: {df['cosine_to_ref'].median():.4f}")
        else:
            print("  No shared vocabulary found!")

    # Generate visualizations only if we have enough data
    pairs = [(sent1_og_model, sent1_out_model, "Sent1"), (sent2_og_model, sent2_out_model, "Sent2")]
    for og_model, out_model, label in pairs:
        try:
            og_vecs, out_vecs, words = prepare_vectors_for_viz(og_model, out_model, args.max_words_viz)
            
            plot_shifts_2d(og_vecs, out_vecs, words, method="pca",
                          outpath=args.outdir / f"shift_pca_{label}.png",
                          title=f"Semantic Shifts: Original vs Reconstructed ({label}) - PCA",
                          max_points=args.max_words_viz)
            
            plot_shifts_2d(og_vecs, out_vecs, words, method="tsne",
                          outpath=args.outdir / f"shift_tsne_{label}.png",
                          title=f"Semantic Shifts: Original vs Reconstructed ({label}) - t-SNE",
                          max_points=args.max_words_viz)
        except Exception as e:
            print(f"Skipping visualization for {label}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Text Word-Shift Analyzer")
    subparsers = parser.add_subparsers(dest='mode', help='Analysis mode')
    
    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--outdir", required=True, type=Path, help="Output directory")
    common_args.add_argument("--vector_size", type=int, default=200, help="Word2Vec vector size")
    common_args.add_argument("--window", type=int, default=5, help="Word2Vec window size")
    common_args.add_argument("--min_count", type=int, default=2, help="Word2Vec min_count")
    common_args.add_argument("--epochs", type=int, default=50, help="Word2Vec epochs")
    common_args.add_argument("--seed", type=int, default=42, help="Random seed")
    common_args.add_argument("--max_words_viz", type=int, default=300, help="Max words to visualize per plot")

    # TXT mode
    txt_parser = subparsers.add_parser('txt', parents=[common_args], help='TXT analysis mode')
    txt_parser.add_argument("--ref", required=True, type=Path, help="Path to reference .txt")
    txt_parser.add_argument("--sysA", required=True, type=Path, help="Path to system A output .txt")
    txt_parser.add_argument("--sysB", required=True, type=Path, help="Path to system B output .txt")
    txt_parser.add_argument("--sysC", required=True, type=Path, help="Path to system C output .txt")

    # SEN mode
    sen_parser = subparsers.add_parser('sen', parents=[common_args], help='SEN analysis mode')
    sen_parser.add_argument("--sent1_og", required=True, type=Path, help="Path to sentence 1 original .txt")
    sen_parser.add_argument("--sent1_out", required=True, type=Path, help="Path to sentence 1 reconstructed .txt")
    sen_parser.add_argument("--sent2_og", required=True, type=Path, help="Path to sentence 2 original .txt")
    sen_parser.add_argument("--sent2_out", required=True, type=Path, help="Path to sentence 2 reconstructed .txt")

    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'txt':
        run_txt_analysis(args)
    elif args.mode == 'sen':
        run_sen_analysis(args)

    print("Analysis complete. Results written to", args.outdir, file=sys.stderr)

if __name__ == "__main__":
    main()