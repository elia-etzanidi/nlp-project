"""
Phase 1 scoring (NO embeddings).

Reads reconstruction outputs from .txt files only (no calls into text_reconstruction.py).

Required filenames in the input directory (default: ./outputs):
  textN_gec.txt        # REQUIRED reference (grammar-only correction)
  textN_pipeA.txt
  textN_pipeB.txt
  textN_pipeC.txt
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import language_tool_python
from language_tool_python.utils import RateLimitError

# ---- shared helpers ----------------------------------------------------------
from util import (
    torch_device,
    clean,
    sent_split,
    distinct_n,
    rouge_l_recall,
    preserves_numbers,
)

# ----------------------- models for Phase 1 -----------------------------------
@dataclass
class Phase1Models:
    lt: language_tool_python.LanguageTool | None
    gpt2_tok: AutoTokenizer
    gpt2: AutoModelForCausalLM

def load_phase1_models() -> Phase1Models:
    lt = None
    try:
        lt = language_tool_python.LanguageTool("en-US")
    except Exception:
        lt = None
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(torch_device())
    return Phase1Models(lt, gpt2_tok, gpt2)

# ----------------------- metrics (non-embedding) ------------------------------
def gpt2_perplexity(model, tok, text: str, stride: int = 1024) -> float:
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    nlls: List[torch.Tensor] = []
    for i in range(0, input_ids.size(1), stride):
        begin, end = i, min(i + stride, input_ids.size(1))
        trg_len = end - begin
        ids_slice = input_ids[:, begin:end]
        target_ids = ids_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(ids_slice, labels=target_ids)
            nlls.append(out.loss * trg_len)
    loss = torch.stack(nlls).sum() / input_ids.size(1)
    return float(torch.exp(loss).item())

def lt_error_count(lt: language_tool_python.LanguageTool | None, text: str) -> int:
    if lt is None:
        return -1
    try:
        return len(lt.check(text))
    except RateLimitError:
        return -1
    except Exception:
        return -1

def score_one_output(gec_text: str, hyp: str, models: Phase1Models) -> Dict:
    hyp = clean(hyp); ref = clean(gec_text)
    lt_errs = lt_error_count(models.lt, hyp)
    return {
        "gpt2_ppl": round(gpt2_perplexity(models.gpt2, models.gpt2_tok, hyp), 2),
        "lt_errors": lt_errs,
        "lt_status": "ok" if lt_errs >= 0 else "skipped",
        "rougeL_recall_vs_gec": round(rouge_l_recall(ref, hyp), 4),
        "length_ratio": round(len(hyp.split()) / max(1, len(ref.split())), 3),
        "sent_count_delta": len(sent_split(hyp)) - len(sent_split(ref)),
        "distinct1": round(distinct_n(hyp, 1), 4),
        "distinct2": round(distinct_n(hyp, 2), 4),
        "numbers_preserved": int(preserves_numbers(ref, hyp)),
    }

# ----------------------- file readers -----------------------------------------
PIPE_SHORT = {"pipeA": "pipeline_A", "pipeB": "pipeline_B", "pipeC": "pipeline_C"}

def _parse_filename(fname: str) -> Tuple[str, str] | None:
    """
    Return (text_id, kind) where kind in {"pipeA","pipeB","pipeC","gec"}.
    Files must be named like: text1_pipeA.txt, text1_gec.txt
    """
    m = re.match(r"^(text\d+?)_(pipeA|pipeB|pipeC|gec)\.txt$", fname)
    if not m:
        return None
    return m.group(1), m.group(2)

def read_folder(indir: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Returns: { text_id: { 'gec': <str>, 'pipeA': <str>?, 'pipeB': <str>?, 'pipeC': <str>? } }
    Only includes files that match the naming convention.
    """
    p = Path(indir)
    store: Dict[str, Dict[str, str]] = {}
    for f in p.glob("*.txt"):
        parsed = _parse_filename(f.name)
        if not parsed:
            continue
        text_id, kind = parsed
        store.setdefault(text_id, {})
        store[text_id][kind] = f.read_text(encoding="utf-8").strip()
    return store

# ----------------------- scoring from files -----------------------------------
def score_reconstruction_folder(indir: str | Path) -> Dict[str, Dict]:
    """
    Requires textN_gec.txt to exist for each textN to be scored.
    """
    models = load_phase1_models()
    files = read_folder(indir)
    scored: Dict[str, Dict] = {}

    for tid, parts in sorted(files.items()):
        ref = parts.get("gec", "").strip()
        if not ref:
            # Skip scoring this text if no GEC reference file
            scored[tid] = {"error": f"Missing required file: {tid}_gec.txt", "skipped": True}
            continue

        per_text = {"gec_reference": ref}
        for short, longname in PIPE_SHORT.items():
            hyp = parts.get(short)
            if not hyp:
                per_text[longname] = {"success": False, "error": f"Missing file: {tid}_{short}.txt"}
                continue
            per_text[longname] = {
                "pipeline": longname,
                "metrics_phase1": score_one_output(ref, hyp, models),
            }

        scored[tid] = per_text

    return scored

# ----------------------- run standalone --------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score reconstructions from .txt files (GEC required).")
    parser.add_argument("--indir", type=str, default="outputs", help="Directory with textN_gec.txt and textN_pipeA/B/C.txt")
    parser.add_argument("--outjson", type=str, default="", help="Optional path to save JSON with scores.")
    args = parser.parse_args()

    scored = score_reconstruction_folder(args.indir)

    for tid, bundle in scored.items():
        print("\n" + "=" * 80)
        print(tid.upper())
        if bundle.get("skipped"):
            print(bundle["error"])
            continue
        print("- GEC reference:\n", bundle["gec_reference"])
        for key in ["pipeline_A", "pipeline_B", "pipeline_C"]:
            entry = bundle.get(key, {})
            if not entry or not entry.get("metrics_phase1"):
                print(f"\n[{key}] FAILED:", entry.get("error", "Unavailable")); continue
            print(f"\n[{entry['pipeline'].upper()}]")
            for k, v in entry["metrics_phase1"].items():
                print(f"  {k}: {v}")

    if args.outjson:
        Path(args.outjson).write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON to: {Path(args.outjson).resolve()}")
