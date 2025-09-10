"""
Phase 1 scoring (NO embeddings).

This file accepts the results dict from text_reconstruction.reconstruct_with_all_pipelines(...)
and computes non-embedding metrics per pipeline.

It is robust to missing 'gec_reference'. If that key is absent, it will recompute a reference
using LanguageTool + T5-GEC from text_reconstruction.
"""

from typing import Dict, List
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import language_tool_python
import torch

# ---- shared helpers (no scoring logic) ---------------------------------------
from util import (
    torch_device,
    clean,
    sent_split,
    distinct_n,
    rouge_l_recall,
    preserves_numbers,
)

# ---- import reconstruction bits just to recompute GEC if needed --------------
# (We only touch LT + T5-GEC; we won't load paraphrasers here.)
from text_reconstruction import (
    load_models as _load_recon_models,
    language_tool_fix as _lt_fix,
    t5_gec as _t5_gec,
)

# ----------------------- models for Phase 1 -----------------------------------
@dataclass
class Phase1Models:
    lt: language_tool_python.LanguageTool
    gpt2_tok: AutoTokenizer
    gpt2: AutoModelForCausalLM

def load_phase1_models() -> Phase1Models:
    try:
        lt = language_tool_python.LanguageTool("en-US")
    except Exception:
        lt = language_tool_python.LanguageToolPublicAPI("en-US")
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

def lt_error_count(lt: language_tool_python.LanguageTool, text: str) -> int:
    try:
        return len(lt.check(text))
    except Exception:
        return 0

def score_one_output(gec_text: str, hyp: str, models: Phase1Models) -> Dict:
    hyp = clean(hyp); ref = clean(gec_text)
    return {
        "gpt2_ppl": round(gpt2_perplexity(models.gpt2, models.gpt2_tok, hyp), 2),
        "lt_errors": lt_error_count(models.lt, hyp),
        "rougeL_recall_vs_gec": round(rouge_l_recall(ref, hyp), 4),
        "length_ratio": round(len(hyp.split()) / max(1, len(ref.split())), 3),
        "sent_count_delta": len(sent_split(hyp)) - len(sent_split(ref)),
        "distinct1": round(distinct_n(hyp, 1), 4),
        "distinct2": round(distinct_n(hyp, 2), 4),
        "numbers_preserved": int(preserves_numbers(ref, hyp)),
    }

# ----------------------- reference resolver -----------------------------------
def _resolve_gec_reference(bundle: Dict) -> str:
    """
    Try to get gec_reference; if missing, recompute GEC from the original text.
    We look for an 'original' string in any pipeline entry.
    """
    # 1) If present, use it.
    if "gec_reference" in bundle and isinstance(bundle["gec_reference"], str):
        return bundle["gec_reference"]

    # 2) Find an original text (any pipeline should carry it).
    original = None
    for key in ("pipeline_A", "pipeline_B", "pipeline_C"):
        entry = bundle.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("original"), str):
            original = entry["original"]
            break
    if not original:
        raise KeyError("Could not find 'gec_reference' or an 'original' text to rebuild the reference.")

    # 3) Recompute reference using text_reconstruction’s LT + T5-GEC.
    recon_models = _load_recon_models()
    ref = _t5_gec(recon_models.gec_pipe, _lt_fix(recon_models.lt, original))
    return ref

# ----------------------- public API ------------------------------------------
def score_reconstruction_results(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Accepts the dict returned by text_reconstruction.reconstruct_with_all_pipelines(...)
    and returns a new dict with Phase-1 metrics per pipeline.

    Robust to missing 'gec_reference' (recomputes it if needed).
    """
    models = load_phase1_models()
    scored: Dict[str, Dict] = {}

    for text_id, bundle in results.items():
        # Resolve or rebuild the GEC reference
        ref = _resolve_gec_reference(bundle)

        per_text = {}
        for key in ["pipeline_A", "pipeline_B", "pipeline_C"]:
            entry = bundle.get(key, {})
            if not entry or not entry.get("success"):
                per_text[key] = {"success": False, "error": entry.get("error", "Unavailable")}
                continue
            hyp = entry.get("reconstructed")
            if not isinstance(hyp, str):
                per_text[key] = {"success": False, "error": "Missing 'reconstructed' text"}
                continue
            per_text[key] = {
                "pipeline": entry.get("pipeline", key),
                "metrics_phase1": score_one_output(ref, hyp, models),
            }

        scored[text_id] = {
            "gec_reference": ref,
            **per_text,
        }

    return scored

# ----------------------- run standalone --------------------------------------
if __name__ == "__main__":
    # Example: run reconstruction first, then score
    from text_reconstruction import reconstruct_with_all_pipelines

    texts = {
        "text1": {"original": "Today is our dragon boat festival, in our Chinese culture, to celebrate it ..."},
        "text2": {"original": "During our final discuss, I told him about the new submission — the one we were waiting ..."},
    }

    recon = reconstruct_with_all_pipelines(texts)
    scored = score_reconstruction_results(recon)

    # Pretty print
    for tid, bundle in scored.items():
        print("\n" + "=" * 80)
        print(tid.upper())
        print("- GEC reference:\n", bundle["gec_reference"])
        for key in ["pipeline_A", "pipeline_B", "pipeline_C"]:
            entry = bundle[key]
            if not entry.get("success", True):
                print(f"\n[{key}] FAILED:", entry.get("error")); continue
            print(f"\n[{entry['pipeline'].upper()}]")
            print(entry["metrics_phase1"])
