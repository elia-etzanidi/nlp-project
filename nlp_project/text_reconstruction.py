from typing import Dict, Callable
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

# ---- small helpers -----------------------------------------------------------

def device():
    return 0 if torch.cuda.is_available() else -1

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---- pipelines ---------------------------------------------------------------

def setup_pipelines() -> Dict[str, Callable]:
    """
    Setup three different NLP pipelines for text reconstruction
    Returns: dict of pipeline callables
    """

    # 1) T5 grammar correction (fine-tuned)
    # Good community model for GEC
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
        out = t5_gc_pipe(
            text.strip(),
            max_new_tokens=128,
            do_sample=False,
            num_beams=4
        )[0]["generated_text"]
        return out.strip()

    # 2) BART paraphrase (fine-tuned)
    bart_para_name = "eugenesiow/bart-paraphrase"
    bart_para = pipeline(
        "text2text-generation",
        model=bart_para_name,
        device=device()
    )

    def bart_reconstruction(text: str) -> str:
        prompt = text.strip()
        out = bart_para(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            num_beams=4
        )[0]["generated_text"]
        return out.strip()

    # 3) FLAN-T5 for instruction-style “improve & correct”
    flan_name = "google/flan-t5-base"
    flan_tok = AutoTokenizer.from_pretrained(flan_name)
    flan = AutoModelForSeq2SeqLM.from_pretrained(flan_name)
    flan_pipe = pipeline(
        "text2text-generation",
        model=flan,
        tokenizer=flan_tok,
        device=device()
    )

    def flan_reconstruction(text: str) -> str:
        prompt = (
            "Improve clarity and grammar. Keep meaning and names. "
            "Rewrite as a polished email paragraph:\n\n" + text.strip()
        )
        out = flan_pipe(
            prompt,
            max_new_tokens=160,
            do_sample=False,
            num_beams=4
        )[0]["generated_text"]
        return out.strip()

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