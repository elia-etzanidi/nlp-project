"""
Phase 1 scoring.

Reads reconstruction outputs from .txt files.

Required filenames in the input directory:
  textN_gec.txt
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

# ------------------------- shared helpers ---------------------------------
from util import (
    torch_device,
    clean,
    sent_split,
    distinct_n,
    rouge_l_recall,
    preserves_numbers,
)

# --------------------------- models ---------------------------------------
@dataclass
class Phase1Models:
    gpt2_tok: AutoTokenizer
    gpt2: AutoModelForCausalLM

def load_phase1_models(model_name: str = "gpt2") -> Phase1Models:
    """Load GPT-2 model and tokenizer for perplexity calculation."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(torch_device())
        return Phase1Models(tokenizer, model)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

# ------------------------------ metrics ------------------------------------
def gpt2_perplexity(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                   text: str, stride: int = 1024) -> float:
    """Calculate perplexity using GPT-2 model."""
    if not text.strip():
        return float('inf')
    
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    
    neg_log_likelihoods: List[torch.Tensor] = []
    
    for i in range(0, input_ids.size(1), stride):
        begin_loc = i
        end_loc = min(i + stride, input_ids.size(1))
        target_len = end_loc - begin_loc
        
        input_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_slice.clone()
        target_ids[:, :-target_len] = -100
        
        with torch.no_grad():
            outputs = model(input_slice, labels=target_ids)
            neg_log_likelihoods.append(outputs.loss * target_len)
    
    perplexity = torch.exp(torch.stack(neg_log_likelihoods).sum() / input_ids.size(1))
    return float(perplexity.item())

def score_one_output(reference_text: str, hypothesis_text: str, models: Phase1Models) -> Dict:
    """Score a single pipeline output against the GEC reference."""
    clean_hypothesis = clean(hypothesis_text)
    clean_reference = clean(reference_text)
    
    return {
        "gpt2_ppl": round(gpt2_perplexity(models.gpt2, models.gpt2_tok, clean_hypothesis), 2),
        "rougeL_recall_vs_gec": round(rouge_l_recall(clean_reference, clean_hypothesis), 4),
        "length_ratio": round(len(clean_hypothesis.split()) / max(1, len(clean_reference.split())), 3),
        "sent_count_delta": len(sent_split(clean_hypothesis)) - len(sent_split(clean_reference)),
        "distinct1": round(distinct_n(clean_hypothesis, 1), 4),
        "distinct2": round(distinct_n(clean_hypothesis, 2), 4),
        "numbers_preserved": int(preserves_numbers(clean_reference, clean_hypothesis)),
    }

# ----------------------- file readers -----------------------------------------
# Expected pipeline types (adjust as needed)
EXPECTED_PIPELINES = {
    "pipeA": "pipeline_A",
    "pipeB": "pipeline_B", 
    "pipeC": "pipeline_C"
}

def _parse_filename(filename: str) -> Tuple[str, str] | None:
    """
    Parse filename to extract text_id and pipeline type.
    
    Expected format: text1_pipeA.txt, text1_pipeB.txt, text1_pipeC.txt, text1_gec.txt
    Returns (text_id, pipeline_type) or None if invalid.
    """
    pattern = r"^(text\d+)_(pipeA|pipeB|pipeC|gec)\.txt$"
    match = re.match(pattern, filename)
    return (match.group(1), match.group(2)) if match else None

def read_folder(input_dir: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Read all valid text files from directory.
    
    Returns: {text_id: {'gec': content, 'pipeB': content, ...}}
    """
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    
    files_by_text: Dict[str, Dict[str, str]] = {}
    
    for file_path in directory.glob("*.txt"):
        parsed = _parse_filename(file_path.name)
        if not parsed:
            continue
            
        text_id, pipeline_type = parsed
        content = file_path.read_text(encoding="utf-8").strip()
        
        files_by_text.setdefault(text_id, {})
        files_by_text[text_id][pipeline_type] = content
    
    return files_by_text

# ----------------------------- scoring ------------------------------------------
def score_reconstruction_folder(input_dir: str | Path) -> Dict[str, Dict]:
    """
    Score all reconstruction outputs in a folder.
    
    Requires textN_gec.txt to exist for each textN to be scored.
    """
    models = load_phase1_models()
    files_data = read_folder(input_dir)
    results: Dict[str, Dict] = {}

    for text_id, file_contents in sorted(files_data.items()):
        gec_reference = file_contents.get("gec", "").strip()
        
        if not gec_reference:
            results[text_id] = {
                "error": f"Missing required file: {text_id}_gec.txt", 
                "skipped": True
            }
            continue

        text_results = {"gec_reference": gec_reference}
        
        for short_name, long_name in EXPECTED_PIPELINES.items():
            pipeline_output = file_contents.get(short_name)
            
            if not pipeline_output:
                text_results[long_name] = {
                    "success": False, 
                    "error": f"Missing file: {text_id}_{short_name}.txt"
                }
                continue
            
            text_results[long_name] = {
                "pipeline": long_name,
                "metrics_phase1": score_one_output(gec_reference, pipeline_output, models),
            }

        results[text_id] = text_results

    return results

# ----------------------------- main ---------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Score text reconstruction outputs against GEC references."
    )
    parser.add_argument(
        "--indir", 
        type=str, 
        default="outputs", 
        help="Directory containing textN_gec.txt and pipeline output files"
    )
    parser.add_argument(
        "--outjson", 
        type=str, 
        default="", 
        help="Optional path to save results as JSON"
    )
    
    args = parser.parse_args()

    try:
        results = score_reconstruction_folder(args.indir)
    except Exception as e:
        print(f"Error processing folder: {e}")
        exit(1)

    # Print results
    for text_id, text_data in results.items():
        print("\n" + "=" * 80)
        print(text_id.upper())
        
        if text_data.get("skipped"):
            print(text_data["error"])
            continue
            
        for pipeline_name in EXPECTED_PIPELINES.values():
            pipeline_data = text_data.get(pipeline_name, {})
            
            if not pipeline_data or not pipeline_data.get("metrics_phase1"):
                error_msg = pipeline_data.get("error", "Unavailable")
                print(f"\n[{pipeline_name}] FAILED: {error_msg}")
                continue
            
            print(f"\n[{pipeline_data['pipeline'].upper()}]")
            for metric_name, metric_value in pipeline_data["metrics_phase1"].items():
                print(f"  {metric_name}: {metric_value}")