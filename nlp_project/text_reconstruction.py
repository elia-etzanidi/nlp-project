from typing import Dict, Callable, List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import unicodedata
import re

# ---- small helpers -----------------------------------------------------------

def device():
    return 0 if torch.cuda.is_available() else -1

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

torch.set_grad_enabled(False)
set_seed(42)

def clean(text: str) -> str:  # <-- added
    x = unicodedata.normalize("NFKC", text)
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\s*\n\s*", " ", x).strip()
    return x

def chunk_by_tokens(text: str, tokenizer, max_src_tokens: int = 448, overlap_tokens: int = 32) -> List[str]:
    """
    Split text into token chunks under max_src_tokens with a small overlap.
    Avoids input truncation inside HF pipeline/generate.
    """
    text = clean(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    i = 0
    while i < len(tokens):
        j = min(i + max_src_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[i:j], skip_special_tokens=True)
        chunks.append(chunk.strip())
        if j == len(tokens):
            break
        i = j - overlap_tokens  # overlap for context continuity
        if i < 0:
            i = 0
    return chunks

def sent_split(x: str):
    # Simple splitter; swap with spaCy if you like
    sents = re.split(r"(?<=[.!?])\s+", x)
    return [s.strip() for s in sents if s.strip()]

def preserves_numbers(src: str, hyp: str) -> bool:
    for n in re.findall(r"\b\d+(?:\.\d+)?\b", src):
        if n not in hyp:
            return False
    return True

# ---- pipelines ---------------------------------------------------------------

def setup_pipelines() -> Dict[str, Callable]:
    """
    Setup three different NLP pipelines for text reconstruction
    Returns: dict of pipeline callables
    """

    # 1) T5 grammar correction (fine-tuned)
    t5_gc_name = "vennify/t5-base-grammar-correction"
    t5_gc_tok = AutoTokenizer.from_pretrained(t5_gc_name)
    t5_gc = AutoModelForSeq2SeqLM.from_pretrained(t5_gc_name)
    t5_gc_pipe = pipeline(
        "text2text-generation",
        model=t5_gc,
        tokenizer=t5_gc_tok,
        device=device()
    )

    def t5_reconstruction(text: str) -> str:
        x = clean(text)
        # Split into sentences so each decode stays short & complete
        sents = re.split(r"(?<=[.!?])\s+", x)
        outputs = []

        for s in sents:
            s = s.strip()
            if not s:
                continue

            # First pass (normal budget)
            out = t5_gc_pipe(
                s,
                num_beams=6,
                early_stopping=False,
                no_repeat_ngram_size=3,
                max_new_tokens=96,      # per-sentence budget
                min_new_tokens=16,
                length_penalty=1.05,
                truncation=True
            )[0]["generated_text"].strip()

            # Fallback: if it looks truncated (no terminal punctuation),
            # give it a bit more budget for this sentence only.
            if not re.search(r"[.!?]['\"]?$", out) and len(s) > 80:
                out = t5_gc_pipe(
                    s,
                    num_beams=8,
                    early_stopping=False,
                    no_repeat_ngram_size=3,
                    max_new_tokens=144,   # slightly larger retry
                    min_new_tokens=24,
                    length_penalty=1.05,
                    truncation=True
                )[0]["generated_text"].strip()

            outputs.append(out)

        return " ".join(outputs)

    # 2) BART paraphrase (fine-tuned)
    bart_para_name = "eugenesiow/bart-paraphrase"
    bart_para = pipeline(
        "text2text-generation",
        model=bart_para_name,
        device=device()
    )

    def bart_reconstruction(text: str) -> str:
        # Make BART do grammar itself (paraphrase + correct)
        out = bart_para(
            clean(text),
            num_beams=6,
            early_stopping=False,
            no_repeat_ngram_size=3,
            max_new_tokens=256,
            min_new_tokens=64,
            length_penalty=1.05,
            truncation=True
        )[0]["generated_text"].strip()
        return out

    # 3) FLAN-T5 for instruction-style “improve & correct”
    flan_name = "google/flan-t5-large"
    flan_tok = AutoTokenizer.from_pretrained(flan_name)
    flan = AutoModelForSeq2SeqLM.from_pretrained(flan_name)
    flan_pipe = pipeline(
        "text2text-generation",
        model=flan,
        tokenizer=flan_tok,
        device=device()
    )

    def flan_reconstruction(text: str) -> str:
        """
        Simple 'more synonyms' paraphraser (medium+ strength).
        One pass per sentence; keeps entities/numbers; no T5 fallback.
        """
        sents = sent_split(text)
        if not sents:
            return clean(text)

        out = []
        for s in sents:
            prompt = (
                "Rewrite the sentence using noticeably different wording. "
                "Use appropriate SYNONYMS for at least ~30% of the content words and vary the structure, "
                "but preserve ALL facts and nuances. "
                "Do NOT summarize or change meaning. "
                "Keep names, numbers, dates, and technical terms exactly as in the original. "
                "Return exactly ONE sentence.\n\n"
                f"Sentence: {s}\nParaphrase:"
            )

            y = flan_pipe(
                prompt,
                num_beams=1,              # avoid beam-length bias
                do_sample=True,           # enable creativity
                temperature=0.95,         # push more variation
                top_p=0.95,               # allow rarer synonyms
                no_repeat_ngram_size=3,
                max_new_tokens=min(256, max(48, int(len(s) * 1.7))),
                min_new_tokens=22,
                length_penalty=1.20,
                early_stopping=False,
            )[0]["generated_text"].strip()

            # quick cleanup + sentence ending
            y = re.sub(r"^\s*(Paraphrase:|Output:)\s*", "", y, flags=re.I).strip()
            y = re.sub(r"^\(\d+\)\s*", "", y).strip()
            if not re.search(r"[.!?]['\"]?$", y):
                y += "."

            # simple safety: if numbers changed, keep original
            out.append(y if preserves_numbers(s, y) else s)

        return " ".join(out)
    
    return {
        "t5_grammar": t5_reconstruction,
        "bart_paraphrase": bart_reconstruction,
        "flan_improve": flan_reconstruction
    }

def apply_pipeline_reconstruction(text: str, pipeline_name: str, pipeline_func: Callable) -> Dict:
    try:
        reconstructed = pipeline_func(text)
        return {
            "pipeline": pipeline_name,
            "original": text,
            "reconstructed": reconstructed,
            "success": True
        }
    except Exception as e:
        return {
            "pipeline": pipeline_name,
            "original": text,
            "reconstructed": text,
            "success": False,
            "error": str(e)
        }

def reconstruct_with_all_pipelines(texts: Dict) -> Dict[str, Dict]:
    pipelines = setup_pipelines()
    results = {}
    for text_id, text_data in texts.items():
        if text_id.startswith("text"):
            original_text = text_data["original"]
            results[text_id] = {}
            for name, fn in pipelines.items():
                results[text_id][name] = apply_pipeline_reconstruction(original_text, name, fn)
    return results

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
        }
    }
    pipeline_results = reconstruct_with_all_pipelines(texts)

    for text_id, results in pipeline_results.items():
        print(f"\n{text_id.upper()}:")
        for pipeline_name, result in results.items():
            if result["success"]:
                print(f"\n[{pipeline_name.upper()} RESULT]")
                print(result["reconstructed"])
            else:
                print(f"\n[{pipeline_name.upper()} FAILED] {result.get('error', 'Unknown error')}")