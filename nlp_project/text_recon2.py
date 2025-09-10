# Models used:
#   GEC:        vennify/t5-base-grammar-correction  (edit-style GEC; robust)
#   Paraphrase: tuner007/pegasus_paraphrase
#               eugenesiow/bart-paraphrase
#   Rewriter:   google/flan-t5-large
#   Similarity: sentence-transformers/all-MiniLM-L6-v2
#   Fluency:    gpt2 (perplexity proxy)
#   Masked LM:  roberta-large (for optional synonym swap)
#
# Notes:
# - No external servers needed. LanguageTool will use public API by default;
#   if you have Java installed, set LT locally via language_tool_python.LanguageToolPublicAPI=False

from typing import Dict, Callable, List, Tuple
from dataclasses import dataclass
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from sentence_transformers import SentenceTransformer, util as sbert_util
import language_tool_python
import torch, math, random, unicodedata, re
from wordfreq import zipf_frequency

# ----------------------- config & seed ----------------------------------------

ENABLE_SYNONYM_SWAP = False  # set True to enable conservative synonym replacements

def device_index():
    return 0 if torch.cuda.is_available() else -1

def torch_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

torch.set_grad_enabled(False)
set_seed(42)

# ----------------------- small helpers ----------------------------------------

def clean(text: str) -> str:
    x = unicodedata.normalize("NFKC", text)
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\s*\n\s*", " ", x).strip()
    return x

def sent_split(x: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", clean(x))
    return [s.strip() for s in sents if s.strip()]

def preserves_numbers(src: str, hyp: str) -> bool:
    for n in re.findall(r"\b\d+(?:\.\d+)?\b", src):
        if n not in hyp:
            return False
    return True

def distinct_n(text: str, n: int = 2) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(grams) / (len(toks)-n+1)

def rouge_l_recall(ref: str, hyp: str) -> float:
    # simple LCS-based ROUGE-L recall approximation
    ref_t = ref.split()
    hyp_t = hyp.split()
    dp = [[0]*(len(hyp_t)+1) for _ in range(len(ref_t)+1)]
    for i in range(1, len(ref_t)+1):
        for j in range(1, len(hyp_t)+1):
            if ref_t[i-1] == hyp_t[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    return lcs / max(1, len(ref_t))

def _token_len(s: str) -> int:
    return len(s.split())

def _within_len_bounds(src: str, hyp: str, lo: float = 0.75, hi: float = 1.35) -> bool:
    sl, hl = max(1, _token_len(src)), max(1, _token_len(hyp))
    return (hl >= max(6, int(sl * lo))) and (hl <= int(sl * hi))   

# ----------------------- models & utilities -----------------------------------

@dataclass
class ModelBundle:
    lt: language_tool_python.LanguageTool
    gec_pipe: Callable
    peg_pipe: Callable
    bart_pipe: Callable
    flan_pipe: Callable
    sbert: SentenceTransformer
    gpt2_tok: AutoTokenizer
    gpt2: AutoModelForCausalLM
    mlm_tok: AutoTokenizer
    mlm: AutoModelForMaskedLM

def load_models() -> ModelBundle:
    # LanguageTool (rule-based grammar checker)
    try:
        lt = language_tool_python.LanguageTool('en-US')
    except Exception:
        # Fallback to public API
        lt = language_tool_python.LanguageToolPublicAPI('en-US')

    # GEC: T5 grammar correction
    t5_gc_name = "vennify/t5-base-grammar-correction"
    t5_gc_tok = AutoTokenizer.from_pretrained(t5_gc_name)
    t5_gc = AutoModelForSeq2SeqLM.from_pretrained(t5_gc_name)
    gec_pipe = pipeline("text2text-generation", model=t5_gc, tokenizer=t5_gc_tok, device=device_index())

    # Paraphrase models
    peg_name = "tuner007/pegasus_paraphrase"
    peg_pipe = pipeline("text2text-generation", model=peg_name, device=device_index())
    bart_name = "eugenesiow/bart-paraphrase"
    bart_pipe = pipeline("text2text-generation", model=bart_name, device=device_index())

    # Instructional rewriter
    flan_name = "google/flan-t5-large"
    flan_tok = AutoTokenizer.from_pretrained(flan_name)
    flan = AutoModelForSeq2SeqLM.from_pretrained(flan_name)
    flan_pipe = pipeline("text2text-generation", model=flan, tokenizer=flan_tok, device=device_index())

    # Similarity model
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(torch_device()))

    # Fluency proxy: GPT-2 perplexity
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(torch_device())

    # Masked LM for synonyms
    mlm_tok = AutoTokenizer.from_pretrained("roberta-large")
    mlm = AutoModelForMaskedLM.from_pretrained("roberta-large").to(torch_device())

    return ModelBundle(lt, gec_pipe, peg_pipe, bart_pipe, flan_pipe, sbert, gpt2_tok, gpt2, mlm_tok, mlm)

# ----------------------- core steps -------------------------------------------

def language_tool_fix(lt: language_tool_python.LanguageTool, text: str) -> str:
    text = clean(text)
    try:
        return lt.correct(text)
    except Exception:
        return text  # if LT unavailable, return original

def t5_gec(gec_pipe, text: str) -> str:
    # sentence-wise to avoid truncation + keep control
    outs = []
    for s in sent_split(text):
        y = gec_pipe(
            s,
            num_beams=6,
            no_repeat_ngram_size=3,
            max_new_tokens=96,
            length_penalty=1.05,
            truncation=True
        )[0]["generated_text"].strip()
        outs.append(y if preserves_numbers(s, y) else s)
    return " ".join(outs)

def paraphrase_pegasus_sentencewise(peg_pipe, text: str) -> str:
    if peg_pipe is None:
        return text
    outs = []
    for s in sent_split(text):
        # Keep PEGASUS within its ~60 token cap
        try:
            res = peg_pipe(
                s,
                num_beams=6,
                do_sample=False,
                no_repeat_ngram_size=3,
                max_length=60,   # <-- important: respect model cap
                min_length=8,    # keep something substantive
                length_penalty=1.1,
                truncation=True,
            )[0]["generated_text"].strip()
        except Exception:
            res = s  # fail-safe: keep original sentence

        # guards: keep numbers & reasonable length; otherwise fall back
        def _token_len(x): return len(x.split())
        def _within_len_bounds(src, hyp, lo=0.75, hi=1.35):
            sl, hl = max(1, _token_len(src)), max(1, _token_len(hyp))
            return (hl >= max(6, int(sl*lo))) and (hl <= int(sl*hi))

        if not preserves_numbers(s, res) or not _within_len_bounds(s, res):
            outs.append(s)
        else:
            outs.append(res)
    return " ".join(outs)

def paraphrase_bart_sentencewise(bart_pipe, text: str) -> str:
    outs = []
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
        # safety guards: keep numbers & reasonable length
        sl, hl = max(1, len(s.split())), max(1, len(y.split()))
        if not preserves_numbers(s, y) or hl < max(6, int(sl*0.75)) or hl > int(sl*1.35):
            outs.append(s)
        else:
            outs.append(y)
    return " ".join(outs)

def rewrite_flan(flan_pipe, text: str) -> str:
    outs = []
    for s in sent_split(text):
        prompt = (
            "Rewrite the sentence to be fluent, natural English while preserving ALL meaning. "
            "Vary the wording and structure. Do not summarize. Keep names and numbers unchanged.\n\n"
            f"Sentence: {s}\nParaphrase:"
        )
        y = flan_pipe(
            prompt,
            do_sample=False,           # determinism for evaluation
            num_beams=4,
            no_repeat_ngram_size=3,
            max_new_tokens=min(192, max(48, int(len(s) * 1.6))),
            length_penalty=1.05,
            truncation=True
        )[0]["generated_text"].strip()
        y = re.sub(r"^\s*(Paraphrase:|Output:)\s*", "", y, flags=re.I).strip()
        outs.append(y if preserves_numbers(s, y) else s)
    return " ".join(outs)

# ----------------------- scoring & reranking ----------------------------------

def gpt2_perplexity(model, tok, text: str, stride: int = 1024) -> float:
    # memory-safe perplexity over long text
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin = i
        end = min(i+stride, input_ids.size(1))
        trg_len = end - begin
        input_ids_slice = input_ids[:, begin:end]
        target_ids = input_ids_slice.clone()
        # mask all but last trg_len tokens
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(input_ids_slice, labels=target_ids)
            nlls.append(out.loss * trg_len)
    loss = torch.stack(nlls).sum() / input_ids.size(1)
    return float(torch.exp(loss).item())

def lt_error_count(lt: language_tool_python.LanguageTool, text: str) -> int:
    try:
        matches = lt.check(text)
        return len(matches)
    except Exception:
        return 0

def sbert_sim(sbert: SentenceTransformer, a: str, b: str) -> float:
    ea, eb = sbert.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(sbert_util.cos_sim(ea, eb).item())

@dataclass
class Scores:
    sim: float
    ppl: float
    lt_errs: int
    rougeL_to_gec: float
    distinct1: float
    distinct2: float
    total: float

def score_candidate(
    candidate: str,
    gec_text: str,
    lt: language_tool_python.LanguageTool,
    sbert: SentenceTransformer,
    gpt2, gpt2_tok
) -> Scores:
    sim = sbert_sim(sbert, candidate, gec_text)
    ppl = gpt2_perplexity(gpt2, gpt2_tok, candidate)
    lt_errs = lt_error_count(lt, candidate)
    rougeL = rouge_l_recall(gec_text, candidate)
    d1 = distinct_n(candidate, 1)
    d2 = distinct_n(candidate, 2)

    # Combine: higher sim, lower ppl, lower lt_errs, lower rouge (more paraphrase) but not too low.
    # We transform to z-ish via simple monotonic maps.
    score = (
        + 2.0 * sim
        - 0.002 * ppl
        - 0.10 * lt_errs
        + 0.50 * (1.0 - rougeL)      # encourage more rewording while preserving sim
        + 0.25 * (d1 + d2)
    )
    return Scores(sim, ppl, lt_errs, rougeL, d1, d2, score)

# ----------------------- controlled synonym swap (optional) -------------------

CONTENT_POS_LIKE = re.compile(r"\b(NOUN|VERB|ADJ|ADV)\b", re.I)

def simple_pos(word: str) -> str:
    # tiny heuristic POS (avoid heavy spaCy/NLTK). Very crude:
    if re.search(r"ly$", word.lower()):
        return "ADV"
    if re.search(r"(ous|ful|ive|able|al|ic|ish|ary|less|est|er)$", word.lower()):
        return "ADJ"
    if re.search(r"(ed|ing|en)$", word.lower()):
        return "VERB"
    return "NOUN"

def masked_lm_synonym_swap(text: str, mlm, mlm_tok, sbert, ref_text: str) -> str:
    # Conservative: try to replace at most 1 content word per sentence with a frequent alternative.
    out_sents = []
    for s in sent_split(text):
        tokens = s.split()
        if len(tokens) < 6:
            out_sents.append(s); continue

        # pick a middle token that looks like content
        candidates_idx = [i for i, w in enumerate(tokens) if re.match(r"^[A-Za-z][A-Za-z\-']+$", w)]
        random.shuffle(candidates_idx)
        replaced = False

        for i in candidates_idx:
            w = tokens[i]
            if w[0].isupper():  # skip names
                continue
            pos_guess = simple_pos(w)
            if not CONTENT_POS_LIKE.match(pos_guess):
                continue
            if zipf_frequency(w.lower(), "en") < 3.5:
                continue  # avoid rare originals

            masked = tokens[:i] + [mlm_tok.mask_token] + tokens[i+1:]
            masked_text = " ".join(masked)
            enc = mlm_tok(masked_text, return_tensors="pt").to(mlm.device)
            with torch.no_grad():
                logits = mlm(**enc).logits
            mask_index = (enc.input_ids[0] == mlm_tok.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_index) == 0:
                continue
            mask_index = mask_index[0].item()
            probs = torch.softmax(logits[0, mask_index], dim=-1)
            topk = torch.topk(probs, k=10)
            cands = [mlm_tok.decode([idx]).strip() for idx in topk.indices.tolist()]

            # choose candidate with similar POS-ish heuristic & frequency
            for c in cands:
                if c.lower() == w.lower():  # no-op
                    continue
                if not re.match(r"^[A-Za-z][A-Za-z\-']+$", c):
                    continue
                if abs(len(c) - len(w)) > 6:
                    continue
                if zipf_frequency(c.lower(), "en") < 3.5:
                    continue
                # quick semantic guard
                trial = " ".join(tokens[:i] + [c] + tokens[i+1:])
                if sbert_sim(sbert, trial, ref_text) >= 0.92:
                    tokens[i] = c
                    replaced = True
                    break
            if replaced:
                break

        out_sents.append(" ".join(tokens))
    return " ".join(out_sents)

# ----------------------- pipelines -------------------------------------------

def pipeline_A(models: ModelBundle, text: str) -> str:
    x0 = language_tool_fix(models.lt, text)
    x1 = t5_gec(models.gec_pipe, x0)
    y  = paraphrase_pegasus_sentencewise(models.peg_pipe, x1)
    return y

def pipeline_B(models: ModelBundle, text: str) -> str:
    # T5-GEC -> FLAN rewrite (instructional)
    x1 = t5_gec(models.gec_pipe, text)
    x2 = rewrite_flan(models.flan_pipe, x1)
    return x2

def pipeline_C(models: ModelBundle, text: str) -> str:
    # Simple: Grammar fix -> BART paraphrase (sentence-wise)
    x0 = language_tool_fix(models.lt, text)
    x1 = t5_gec(models.gec_pipe, x0)

    y = paraphrase_bart_sentencewise(models.bart_pipe, x1)

    # Optional extra guard (keeps 1:1 sentence mapping with the GEC reference)
    # y = enforce_sentence_count(x1, y)

    return y

# ----------------------- runner & comparison ----------------------------------

def apply_pipeline(name: str, fn: Callable[[ModelBundle, str], str], models: ModelBundle, text: str) -> Dict:
    try:
        y = fn(models, text)
        return {"pipeline": name, "original": text, "reconstructed": y, "success": True}
    except Exception as e:
        return {"pipeline": name, "original": text, "reconstructed": text, "success": False, "error": str(e)}

def evaluate_against_gec(models: ModelBundle, gec_text: str, hyp: str) -> Dict:
    return {
        "sbert_sim_to_gec": round(sbert_sim(models.sbert, hyp, gec_text), 4),
        "gpt2_ppl": round(gpt2_perplexity(models.gpt2, models.gpt2_tok, hyp), 2),
        "lt_errors": lt_error_count(models.lt, hyp),
        "rougeL_recall_vs_gec": round(rouge_l_recall(gec_text, hyp), 4),
        "distinct1": round(distinct_n(hyp, 1), 4),
        "distinct2": round(distinct_n(hyp, 2), 4),
    }

def reconstruct_with_all_pipelines(texts: Dict[str, Dict]) -> Dict[str, Dict]:
    models = load_models()
    results: Dict[str, Dict] = {}
    for text_id, text_data in texts.items():
        if not text_id.startswith("text"):
            continue
        original = text_data["original"]

        # shared GEC reference for fair comparison of meaning/similarity
        gec_ref = t5_gec(models.gec_pipe, language_tool_fix(models.lt, original))

        pA = apply_pipeline("pipeline_A_gec→pegasus(+syn)", pipeline_A, models, original)
        pB = apply_pipeline("pipeline_B_gec→flan", pipeline_B, models, original)
        pC = apply_pipeline("pipeline_C_gec→bart", pipeline_C, models, original)

        # attach metrics relative to gec_ref
        for p in [pA, pB, pC]:
            if p["success"]:
                p["metrics"] = evaluate_against_gec(models, gec_ref, p["reconstructed"])
            else:
                p["metrics"] = None

        results[text_id] = {
            "gec_reference": gec_ref,
            "pipeline_A": pA,
            "pipeline_B": pB,
            "pipeline_C": pC,
        }
    return results

# ----------------------- demo / main ------------------------------------------

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

    results = reconstruct_with_all_pipelines(texts)

    # pretty print
    for tid, bundle in results.items():
        print("\n" + "="*80)
        print(tid.upper())
        print("- GEC reference:\n", bundle["gec_reference"])

        for key in ["pipeline_A", "pipeline_B", "pipeline_C"]:
            res = bundle[key]
            print(f"\n[{res['pipeline'].upper()}]")
            if res["success"]:
                print(res["reconstructed"])
                print("metrics:", res["metrics"])
            else:
                print("FAILED:", res.get("error", "Unknown error"))