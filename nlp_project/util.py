from typing import List
import torch, random, unicodedata, re

# ----------------------- seed & device helpers --------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_index() -> int:
    return 0 if torch.cuda.is_available() else -1

def torch_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def torch_device_str() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

# ----------------------- text helpers -----------------------------------------
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

def within_len_bounds(src: str, hyp: str, lo: float = 0.75, hi: float = 1.35) -> bool:
    sl, hl = max(1, len(src.split())), max(1, len(hyp.split()))
    return (hl >= max(6, int(sl * lo))) and (hl <= int(sl * hi))