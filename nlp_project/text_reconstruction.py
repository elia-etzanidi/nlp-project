"""
text_reconstruction.py
----------------------
Three sentence-by-sentence reconstruction pipelines (no scoring).

Pipelines:
- A: LanguageTool -> T5-GEC -> PEGASUS paraphrase (sentence-wise, length-capped)
- B: T5-GEC -> FLAN-T5 rewrite (instructional, sentence-wise)
- C: LanguageTool -> T5-GEC -> BART paraphrase (sentence-wise)

Dependencies:
  transformers, language-tool-python, torch
Shared helpers imported from util.py:
  set_seed, device_index, clean, sent_split, preserves_numbers, within_len_bounds
"""

from typing import Dict, Callable, List
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import language_tool_python
import torch, re

from util import (
    set_seed,
    device_index,
    clean,
    sent_split,
    preserves_numbers,
    within_len_bounds,
)

# ----------------------- init -------------------------------------------------
torch.set_grad_enabled(False)
set_seed(42)

# ----------------------- models ----------------------------------------------
@dataclass
class ModelBundle:
    lt: language_tool_python.LanguageTool
    gec_pipe: Callable
    peg_pipe: Callable
    bart_pipe: Callable
    flan_pipe: Callable

def load_models() -> ModelBundle:
    # LanguageTool (use local if available; else public API)
    try:
        lt = language_tool_python.LanguageTool("en-US")
    except Exception:
        lt = language_tool_python.LanguageToolPublicAPI("en-US")

    # T5-GEC
    t5_gc_name = "vennify/t5-base-grammar-correction"
    t5_tok = AutoTokenizer.from_pretrained(t5_gc_name)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_gc_name)
    gec_pipe = pipeline("text2text-generation", model=t5, tokenizer=t5_tok, device=device_index())

    # PEGASUS paraphrase (slow tokenizer to avoid Windows/tiktoken issues)
    peg_name = "tuner007/pegasus_paraphrase"
    peg_tok = AutoTokenizer.from_pretrained(peg_name, use_fast=False)
    peg = AutoModelForSeq2SeqLM.from_pretrained(peg_name)
    peg_pipe = pipeline("text2text-generation", model=peg, tokenizer=peg_tok, device=device_index())

    # BART paraphrase
    bart_name = "eugenesiow/bart-paraphrase"
    bart_pipe = pipeline("text2text-generation", model=bart_name, device=device_index())

    # FLAN-T5 rewriter
    flan_name = "google/flan-t5-large"
    flan_tok = AutoTokenizer.from_pretrained(flan_name)
    flan = AutoModelForSeq2SeqLM.from_pretrained(flan_name)
    flan_pipe = pipeline("text2text-generation", model=flan, tokenizer=flan_tok, device=device_index())

    return ModelBundle(lt, gec_pipe, peg_pipe, bart_pipe, flan_pipe)

# ----------------------- core steps (no scoring) ------------------------------
def language_tool_fix(lt: language_tool_python.LanguageTool, text: str) -> str:
    text = clean(text)
    try:
        return lt.correct(text)
    except Exception:
        return text

def t5_gec(gec_pipe, text: str) -> str:
    """Grammar-only correction, sentence-wise."""
    outs: List[str] = []
    for s in sent_split(text):
        y = gec_pipe(
            s,
            num_beams=6,
            no_repeat_ngram_size=3,
            max_new_tokens=96,
            length_penalty=1.05,
            truncation=True,
        )[0]["generated_text"].strip()
        outs.append(y if preserves_numbers(s, y) else s)
    return " ".join(outs)

def paraphrase_pegasus_sentencewise(peg_pipe, text: str) -> str:
    """PEGASUS paraphrase per sentence; hard cap ~60 tokens to avoid warnings."""
    outs: List[str] = []
    for s in sent_split(text):
        try:
            y = peg_pipe(
                s,
                num_beams=6,
                do_sample=False,
                no_repeat_ngram_size=3,
                max_length=60,   # respect pegasus cap
                min_length=8,
                length_penalty=1.1,
                truncation=True,
            )[0]["generated_text"].strip()
        except Exception:
            y = s
        ok = preserves_numbers(s, y) and within_len_bounds(s, y, lo=0.75, hi=1.35)
        outs.append(y if ok else s)
    return " ".join(outs)

def paraphrase_bart_sentencewise(bart_pipe, text: str) -> str:
    """BART paraphrase per sentence; a bit more room than PEGASUS."""
    outs: List[str] = []
    for s in sent_split(text):
        try:
            y = bart_pipe(
                s,
                num_beams=8,
                do_sample=False,
                no_repeat_ngram_size=3,
                max_new_tokens=min(128, max(32, int(len(s) * 1.4))),
                min_new_tokens=max(8, int(len(s) * 0.5)),
                length_penalty=1.15,
                truncation=True,
            )[0]["generated_text"].strip()
        except Exception:
            y = s
        ok = preserves_numbers(s, y) and within_len_bounds(s, y, lo=0.75, hi=1.35)
        outs.append(y if ok else s)
    return " ".join(outs)

def rewrite_flan(flan_pipe, text: str) -> str:
    """FLAN rewriter per sentence with instruction for fluency + paraphrase."""
    outs: List[str] = []
    for s in sent_split(text):
        prompt = (
            "Rewrite the sentence to be fluent, natural English while preserving ALL meaning. "
            "Vary the wording and structure. Do not summarize. Keep names and numbers unchanged.\n\n"
            f"Sentence: {s}\nParaphrase:"
        )
        y = flan_pipe(
            prompt,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            max_new_tokens=min(192, max(48, int(len(s) * 1.6))),
            length_penalty=1.05,
            truncation=True,
        )[0]["generated_text"].strip()
        y = re.sub(r"^\s*(Paraphrase:|Output:)\s*", "", y, flags=re.I).strip()
        outs.append(y if preserves_numbers(s, y) else s)
    return " ".join(outs)

# ----------------------- pipelines -------------------------------------------
def pipeline_A(models: ModelBundle, text: str) -> str:
    """LT -> T5-GEC -> PEGASUS paraphrase (sentence-wise)."""
    x0 = language_tool_fix(models.lt, text)
    x1 = t5_gec(models.gec_pipe, x0)
    return paraphrase_pegasus_sentencewise(models.peg_pipe, x1)

def pipeline_B(models: ModelBundle, text: str) -> str:
    """T5-GEC -> FLAN rewrite (sentence-wise)."""
    x1 = t5_gec(models.gec_pipe, text)
    return rewrite_flan(models.flan_pipe, x1)

def pipeline_C(models: ModelBundle, text: str) -> str:
    """LT -> T5-GEC -> BART paraphrase (sentence-wise)."""
    x0 = language_tool_fix(models.lt, text)
    x1 = t5_gec(models.gec_pipe, x0)
    return paraphrase_bart_sentencewise(models.bart_pipe, x1)

# ----------------------- public API ------------------------------------------
def reconstruct_with_all_pipelines(texts: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Returns per text_id:
      - 'gec_reference': grammar-only baseline
      - 'pipeline_A'/'pipeline_B'/'pipeline_C': { pipeline, original, reconstructed, success, error? }
    """
    models = load_models()
    results: Dict[str, Dict] = {}
    for text_id, text_data in texts.items():
        if not text_id.startswith("text"):
            continue
        original = text_data["original"]
        gec_ref = t5_gec(models.gec_pipe, language_tool_fix(models.lt, original))

        def _apply(name, fn):
            try:
                y = fn(models, original)
                return {"pipeline": name, "original": original, "reconstructed": y, "success": True}
            except Exception as e:
                return {"pipeline": name, "original": original, "reconstructed": original, "success": False, "error": str(e)}

        results[text_id] = {
            "gec_reference": gec_ref,
            "pipeline_A": _apply("pipeline_A_gec→pegasus", pipeline_A),
            "pipeline_B": _apply("pipeline_B_gec→flan",    pipeline_B),
            "pipeline_C": _apply("pipeline_C_gec→bart",    pipeline_C),
        }
    return results

# ----------------------- run alone -------------------------------------------
if __name__ == "__main__":
    texts = {
        "text1": {
            "original": """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all
            safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show
            our words to the doctor, as his next contract checking, to all of us. I got this message to see the
            approved message. In fact, I have received the message from the professor, to show me, this, a couple
            of days ago. I am very appreciated the full support of the professor, for our Springer proceedings
            publication."""
        },
        "text2": {
            "original": """During our final discuss, I told him about the new submission — the one we were waiting
            since last autumn, but the updates was confusing as it not included the full feedback from reviewer or
            maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they
            really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
            and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the
            doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that
            part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate
            the outcome with strong coffee and future targets."""
        },
    }

    out = reconstruct_with_all_pipelines(texts)

    # pretty print (no metrics)
    for tid, bundle in out.items():
        print("\n" + "="*80)
        print(tid.upper())
        print("- GEC reference:\n", bundle["gec_reference"])
        for key in ["pipeline_A", "pipeline_B", "pipeline_C"]:
            res = bundle[key]
            print(f"\n[{res['pipeline'].upper()}]")
            if res["success"]:
                print(res["reconstructed"])
            else:
                print("FAILED:", res.get("error", "Unknown error"))
