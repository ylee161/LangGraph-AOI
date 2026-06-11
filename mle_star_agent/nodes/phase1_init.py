"""nodes/phase1_init.py — Phase 1: Initialization (MLE-STAR Section 3).

Implements all Phase 1 sub-steps as a single LangGraph node:

  3.1.1  Skip check + state restore from CKPT_L0 when both L0 and scores exist
  3.1.2  Data split — grouped board-level 70/15/15 via data_split.py
  3.1.3  Retriever — web search (Tavily → Serper → DuckDuckGo) for M=4 candidate architectures
  3.1.4  Baseline coder — LLM generates one PyTorch training script per candidate
  3.1.5  Candidate evaluator — smoke-run, metric guard, arch ban, best-first sort
  3.1.6  Merger — iterative LLM-driven code-merge to attempt to beat the top candidate
  3.1.7  Save L0 — write CKPT_L0, CKPT_CANDIDATE_SCRIPTS, CKPT_CANDIDATE_SCORES
"""

from __future__ import annotations

import html
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from mle_star_agent import config
from mle_star_agent.guards.code_validator import validate_script
from mle_star_agent.shared import code_runner, metric_guard
from mle_star_agent.shared.acceptance_scoring import is_acceptance_improvement
from mle_star_agent.shared.aoi_smoke_triage import build_smoke_diagnostics, select_full_run_slots
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint, save_checkpoint
from mle_star_agent.shared.knowledge_base import load_kb_from_disk
from mle_star_agent.shared.data_split import build_data_split
from mle_star_agent.shared.llm import build_messages, call_llm, call_llm_json, parse_json_response
from mle_star_agent.shared.metrics_parser import metrics_to_dict, parse_metrics
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 1 constants
# ---------------------------------------------------------------------------

_M = 4  # MLE-STAR Section 3.1: retrieve exactly M=4 candidates
_SEARCH_TIMEOUT = 20.0
_MAX_RESULTS = 6
_FETCH_RESULTS = 9
_RAW_CONTENT_CHARS = 700
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_AUTHORITATIVE_DOMAINS = (
    "arxiv.org", "paperswithcode.com", "openreview.net",
    "proceedings.mlr.press", "pytorch.org", "docs.pytorch.org",
    "huggingface.co", "github.com", "timm.fast.ai",
    "pmc.ncbi.nlm.nih.gov", "nature.com", "ieeexplore.ieee.org",
)

_HARD_EXCLUDED = [t.lower() for t in getattr(config, "HARD_EXCLUDED_ARCHITECTURES", ["convnext", "deit"])]


# ---------------------------------------------------------------------------
# §3.1.3 — Web search helpers (Tavily → Serper → DuckDuckGo)
# ---------------------------------------------------------------------------

def _domain(url: str) -> str:
    m = re.match(r"\s*https?://([^/]+)", url or "")
    return m.group(1).lower().removeprefix("www.") if m else ""


def _authority_rank(url: str) -> int:
    d = _domain(url)
    for i, auth in enumerate(_AUTHORITATIVE_DOMAINS):
        if d == auth or d.endswith("." + auth) or auth in d:
            return i
    return len(_AUTHORITATIVE_DOMAINS)


def _prioritize_and_trim(results: list[dict], limit: int = _MAX_RESULTS) -> list[dict]:
    ranked = sorted(results, key=lambda r: _authority_rank(r.get("url", "")))
    seen: set[str] = set()
    first_per_domain: list[dict] = []
    repeats: list[dict] = []
    for r in ranked:
        d = _domain(r.get("url", ""))
        if d and d not in seen:
            seen.add(d)
            first_per_domain.append(r)
        else:
            repeats.append(r)
    return (first_per_domain + repeats)[:limit]


def _search_tavily(query: str, api_key: str) -> list[dict]:
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": _FETCH_RESULTS,
            "search_depth": "advanced",
            "include_raw_content": True,
        },
        timeout=_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
            "url": r.get("url", ""),
            "raw": (r.get("raw_content") or "")[:_RAW_CONTENT_CHARS],
        }
        for r in data.get("results", [])
    ]


def _search_serper(query: str, api_key: str) -> list[dict]:
    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": _FETCH_RESULTS},
        timeout=_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", ""), "raw": ""}
        for r in data.get("organic", [])[:_FETCH_RESULTS]
    ]


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _search_duckduckgo(query: str) -> list[dict]:
    resp = httpx.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": _USER_AGENT},
        timeout=_SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    text = resp.text
    results: list[dict] = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
        results.append({"title": _strip_tags(m.group(2)), "snippet": "", "url": m.group(1), "raw": ""})
        if len(results) >= _FETCH_RESULTS:
            break
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', text, re.S)
    for i, s in enumerate(snippets[: len(results)]):
        results[i]["snippet"] = _strip_tags(s)
    return results


def web_search(query: str) -> str:
    """Run one web search; returns formatted results string.

    Falls back gracefully through Tavily → Serper → DuckDuckGo → SEARCH_UNAVAILABLE.
    """
    try:
        tavily = os.environ.get("TAVILY_API_KEY")
        serper = os.environ.get("SERPER_API_KEY")
        if tavily:
            results, source = _search_tavily(query, tavily), "tavily"
        elif serper:
            results, source = _search_serper(query, serper), "serper"
        else:
            results, source = _search_duckduckgo(query), "duckduckgo"
    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return (
            f"SEARCH_UNAVAILABLE for query {query!r} ({type(exc).__name__}). "
            "Use knowledge of CURRENT (2024-2025) small-data image-classification backbones "
            "(EfficientNet-B1/B2, MobileNetV3, ResNet-50, DINOv2/CLIP frozen). "
            "Do NOT default to ResNet18-only."
        )

    if not results:
        return (
            f"SEARCH_EMPTY for query {query!r}. No results returned. "
            "Use current small-data backbones."
        )

    results = _prioritize_and_trim(results, _MAX_RESULTS)
    logger.info("web_search via %s for %r — %d result(s)", source, query, len(results))

    lines = [f"SEARCH RESULTS ({source}) for: {query}"]
    for i, r in enumerate(results, 1):
        snippet = r.get("snippet", "").strip()
        raw = re.sub(r"\s+", " ", r.get("raw", "").strip())
        lines.append(f"[{i}] {r.get('title', '').strip()}")
        if snippet:
            lines.append(f"    {snippet}")
        if raw:
            lines.append(f"    PAGE EXCERPT: {raw}")
        lines.append(f"    {r.get('url', '').strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §3.1.3 — Candidate retriever (web search → LLM extraction)
# ---------------------------------------------------------------------------

def _retrieve_candidates(data_split: dict, token_state: dict) -> list[dict]:
    """Return M=4 candidate architecture dicts {model_name, description, example_code}.

    Runs 4 dataset-aware web searches, then calls the LLM to extraction candidates.
    Falls back to LLM knowledge when searches fail.
    """
    stats = data_split.get("stats", {})
    train_size = stats.get("train_size", "~300")
    ng_count = stats.get("ng_count", "?")
    g_count = stats.get("g_count", "?")
    modality = data_split.get("metadata", {}).get("input_modality", "stereo")

    queries = [
        f"pretrained PyTorch image classification {train_size} training samples transfer learning overfitting prevention",
        f"DINOv2 SigLIP CLIP frozen foundation model features linear probe small dataset binary defect pass/fail classification",
        f"PyTorch {modality} surface defect detection industrial inspection pretrained backbone small data DINOv2 vision foundation model",
        f"LoRA AdaptFormer PEFT parameter-efficient fine-tuning vision transformer {train_size} samples timm torchvision example code",
    ]

    search_results: list[str] = []
    for q in queries:
        result = web_search(q)
        search_results.append(result)
        time.sleep(0.5)

    all_results = "\n\n---\n\n".join(search_results)

    system_prompt = (
        "You are an ML expert extracting PyTorch model candidates from web search results.\n"
        "Return a JSON array of exactly 4 objects. Each object must have keys:\n"
        "  model_name: str  — architecture name (e.g. 'EfficientNet-B1')\n"
        "  description: str — 1-2 sentences on why it fits a small-data binary defect task\n"
        "  example_code: str — minimal PyTorch snippet loading/instantiating with pretrained weights\n\n"
        "Diversity requirements:\n"
        "- Include at least ONE lightweight option (<10M params: EfficientNet-B0/B1/B2, MobileNetV3-Large, ResNet-50)\n"
        "- Include at least ONE FROZEN foundation backbone (DINOv2, SigLIP, CLIP) used as frozen feature extractor\n"
        "- Span a range of capacities/adaptation strategies\n"
        "HARD EXCLUSIONS — never return these:\n"
        "- ConvNeXt (any variant) — layernorm stem breaks on 9-channel adapted input\n"
        "- DeiT (any variant) — patch embedding adaptation causes recall collapse\n"
        "Return ONLY the JSON array, no other text."
    )

    user_prompt = (
        f"Task profile: {train_size} training samples, NG={ng_count}, G={g_count}, modality={modality}.\n\n"
        f"Search results:\n\n{all_results}\n\n"
        "Extract the 4 best, diverse, current (2024-2025) PyTorch model candidates from these results.\n"
        "If search results are unavailable, use your knowledge of current small-data backbones.\n"
        "Return JSON array of 4 objects with keys: model_name, description, example_code."
    )

    try:
        candidates = call_llm_json(
            build_messages(system_prompt, user_prompt),
            model=config.MODEL_PRO,
            max_tokens=4096,
            temperature=0.3,
            token_state=token_state,
        )
        if isinstance(candidates, list) and len(candidates) >= 1:
            valid = [c for c in candidates if isinstance(c, dict) and c.get("model_name")]
            if valid:
                return valid[:_M]
    except Exception as exc:
        logger.warning("Candidate retrieval LLM call failed: %s", exc)

    # Hard fallback: return known-good backbones from LLM knowledge
    logger.info("Falling back to built-in candidate list.")
    return [
        {
            "model_name": "EfficientNet-B1",
            "description": "Lightweight pretrained CNN. ~7M params. Efficient on small datasets with standard fine-tuning.",
            "example_code": (
                "from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights\n"
                "backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)\n"
                "backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, 1)"
            ),
        },
        {
            "model_name": "MobileNetV3-Large",
            "description": "Very lightweight pretrained CNN. ~5M params. Robust on small datasets.",
            "example_code": (
                "from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights\n"
                "backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)\n"
                "backbone.classifier[3] = nn.Linear(backbone.classifier[3].in_features, 1)"
            ),
        },
        {
            "model_name": "DINOv2-ViT-S/14 (frozen feature extractor)",
            "description": "Frozen DINOv2 ViT-S/14 as Siamese feature extractor. Exceptional few-shot transfer. No patch embedding modification needed.",
            "example_code": (
                "import torch\n"
                "dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')\n"
                "for p in dinov2.parameters(): p.requires_grad_(False)  # frozen\n"
                "feat_dim = 384  # ViT-S/14 output dim"
            ),
        },
        {
            "model_name": "ResNet-50 (partial unfreeze)",
            "description": "ResNet-50 with frozen early layers and unfrozen layer4. Reliable baseline for small industrial datasets.",
            "example_code": (
                "from torchvision.models import resnet50, ResNet50_Weights\n"
                "backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)\n"
                "for name, p in backbone.named_parameters():\n"
                "    p.requires_grad_(name.startswith('layer4') or name.startswith('fc'))\n"
                "backbone.fc = nn.Linear(2048, 1)"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# §3.1.4 — Baseline coder (LLM generates training scripts)
# ---------------------------------------------------------------------------

_DATA_SPLIT_PATH = str(config.CKPT_DATA_SPLIT)
_MISS_BUDGET = config.MISS_RATE_RELAXED_MAX
_OVERKILL_BUDGET = config.OVERKILL_RELAXED_MAX
_THRESHOLD_MIN = config.THRESHOLD_MIN
_THRESHOLD_MAX = config.THRESHOLD_MAX
_THRESHOLD_STEP = config.THRESHOLD_STEP


def _modality_loading_pattern(modality: str) -> str:
    if modality == "stereo":
        return (
            "Load img_l and img_r (absolute paths from split). "
            "Compute diff = abs(img_l - img_r). Concatenate -> 9-channel input. "
            "Apply identical geometric transforms to both images."
        )
    return (
        "Load img (absolute path from split). Standard 3-channel input. "
        "Standard torchvision transforms, no paired-sync requirement."
    )


def _build_script_gen_prompt(candidate: dict, modality: str, token_state: dict) -> str:
    """Return the system prompt for script generation (used once per candidate)."""
    model_name = candidate.get("model_name", "unknown")
    description = candidate.get("description", "")
    example_code = candidate.get("example_code", "")
    modality_pattern = _modality_loading_pattern(modality)

    return f"""You are an expert PyTorch engineer generating a complete, self-contained AOI training script.

## Candidate architecture
Model: {model_name}
Description: {description}
Example code:
```python
{example_code}
```

## Data loading pattern ({modality})
{modality_pattern}

## MANDATORY requirements (every script MUST follow these)

### Data access
```python
import json
DATA_SPLIT_PATH = "{_DATA_SPLIT_PATH}"
with open(DATA_SPLIT_PATH) as f:
    data_split = json.load(f)
train_samples = data_split["train"]
val_samples   = data_split["val"]
test_samples  = data_split["test"]
```
Each sample has: sample_id (str), img_l + img_r (absolute paths, stereo) or img (mono), label ("G" or "NG").

### Dry-run support (REQUIRED — validator will reject without this)
```python
import os
DRY_RUN         = os.getenv("DRY_RUN") == "1"
DRY_RUN_EPOCHS  = int(os.getenv("DRY_RUN_EPOCHS", "1"))
DRY_RUN_SAMPLES = int(os.getenv("DRY_RUN_SAMPLES", "10"))
```
- When DRY_RUN=1: cap each split to DRY_RUN_SAMPLES, set epochs = DRY_RUN_EPOCHS. Still print METRICS:.
- Full-run epoch count MUST be set via: `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20`

### Device selection (REQUIRED — target machine is Apple Silicon, no CUDA)
Select the device with CUDA → MPS → CPU priority, and move BOTH the model and every
input/label tensor to it:
```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
```
- Use float32 tensors (MPS does not support float64); do not call `.double()` or pass `dtype=torch.float64`.
- Unsupported MPS ops auto-fall back to CPU (PYTORCH_ENABLE_MPS_FALLBACK=1 is set by the runner), so prefer MPS — do NOT hard-code `.cpu()` for training.

### Stereo 9-channel input (stereo modality only)
Load img_l AND img_r, compute abs difference, concatenate -> 9-channel tensor.
When adapting a pretrained 3-channel backbone to 9 channels, initialise the new first conv by repeating pretrained weights / 3.0 across all 3 groups. Apply identical augmentations to L and R.

### ViT/transformer backbones: use feature-level Siamese difference
If using ViT, DINOv2, CLIP, SigLIP or any patch-embedding backbone:
- Do NOT modify the patch embedding for 9 channels
- Use a SHARED encoder: f_L = encoder(img_l), f_R = encoder(img_r)
- Head receives concat([f_L, f_R, abs(f_L-f_R)]) — 3x feature_dim
- Mark: FEATURE_DIFF_CANDIDATE = True near the top

### Labels and loss
Binary: G -> 0, NG -> 1.
```python
n_ng = sum(1 for s in train_samples if s["label"] == "NG")
n_g  = sum(1 for s in train_samples if s["label"] == "G")
pos_weight = torch.tensor([n_g / n_ng])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
```

### LR schedule (MANDATORY — validator hard-rejects without scheduler.step())
Use CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6) OR ReduceLROnPlateau.
Call scheduler.step() in the training loop.

### Layer-wise LR (stereo 9-channel CNN path only)
If a new 9-channel first conv is created, use two param groups: stem at BASE_LR, backbone at BASE_LR/10.
Build the optimizer AFTER assigning new_conv back into the model AND after model.to(device).

### Threshold sweep and calibration
After training:
1. Fit sklearn isotonic regression on val set (X=raw_val_ng_scores, y=binary labels).
2. Map all scores through calibrator.
3. Sweep threshold 0.01..0.99 in steps of 0.01 on calibrated val probabilities.
4. Stage 0: filter FP <= 2; Stage 1: minimise miss_rate; Stage 2: minimise overkill_rate.
5. Report test metrics using calibrated probabilities at the selected threshold.

### Required output markers (MUST all appear in stdout)
1. Pre-training probe: `PROBE_METRICS: {{"ng_recall": X, "overkill_rate": X, "G_prob_mean": X, "NG_prob_mean": X, "should_continue": true/false, "reason": "..."}}`
2. Per-epoch log: `EPOCH_LOG: {{"epoch": N, "train_loss": X, "val_loss": X, "val_ng_recall": X, "val_overkill": X}}`
3. Final metrics: `METRICS: {{"accuracy": X, "ng_recall": X, "miss_rate": X, "overkill_rate": X, "f1": X, "avg_latency_ms": X, "threshold": X, "ng_count": N, "g_count": N, "tp": N, "tn": N, "fp": N, "fn": N, "roc_auc": X, "prob_gap": X}}`
4. Calibration stats: `CALIBRATION_STATS: {{"G_prob_mean": X, "G_prob_std": X, "NG_prob_mean": X, "NG_prob_std": X}}`
5. Threshold curve: `THRESHOLD_CURVE: [{{"t": X, "recall": X, "overkill": X, "miss_rate": X, "accuracy": X}}, ...]`
6. Per-sample predictions: `PREDICTIONS: [{{"sample_id": "...", "true_label": "G", "predicted_label": "NG", "ng_probability": X, "threshold": X}}, ...]`

### Metric definitions
- TP=true NG predicted NG; TN=true G predicted G; FP=true G predicted NG; FN=true NG predicted G
- accuracy=(TP+TN)/(TP+TN+FP+FN); ng_recall=TP/(TP+FN) [1.0 if no NG]; miss_rate=FN/(TP+FN)
- overkill_rate=FP/(TN+FP) [0.0 if no G]; f1=2*P*R/(P+R); prob_gap=mean(NG probs)-mean(G probs)
- roc_auc: sklearn.metrics.roc_auc_score on test (0.0 if single class)

### Small-data policy
- Prefer freeze/partial-freeze + small head, weight decay, AOI-safe augmentation
- Full-run epochs: exactly 20 (via DRY_RUN ternary), early stopping patience=3
- Error handling: wrap training in try/except

Return the COMPLETE Python script as a JSON object: {{"script": "...", "architecture": "{model_name}"}}.
The script must be fully self-contained and runnable as `python script.py`.
"""


def _generate_script(candidate: dict, modality: str, token_state: dict) -> Optional[str]:
    """Call LLM to generate one complete training script for the given candidate."""
    model_name = candidate.get("model_name", "unknown")
    logger.info("Generating script for: %s", model_name)

    system_prompt = _build_script_gen_prompt(candidate, modality, token_state)
    user_prompt = (
        f"Generate the complete self-contained PyTorch training script for: {model_name}.\n"
        "Follow ALL mandatory requirements exactly. "
        "Return JSON: {\"script\": \"...\", \"architecture\": \"" + model_name + "\"}.\n"
        "The script field must contain the FULL Python source code."
    )

    try:
        response = call_llm_json(
            build_messages(system_prompt, user_prompt),
            model=config.MODEL_PRO,
            max_tokens=8192,
            temperature=0.2,
            token_state=token_state,
        )
        if isinstance(response, dict):
            script = response.get("script", "")
            if script and len(script) > 200:
                return script
    except Exception as exc:
        logger.warning("Script generation failed for %s: %s", model_name, exc)

    return None


# ---------------------------------------------------------------------------
# §3.1.5 — Candidate evaluator helpers
# ---------------------------------------------------------------------------

def _norm_arch(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_excluded(name: str, arch: str) -> bool:
    key = f"{name} {arch}".lower()
    return any(t in key for t in _HARD_EXCLUDED)


def _load_runtime_failed_arch_terms() -> list[str]:
    if not checkpoint_exists(config.CKPT_FAILED_ARCHITECTURES):
        return []
    try:
        data = load_checkpoint(config.CKPT_FAILED_ARCHITECTURES)
        terms = []
        for e in data.get("failed", []):
            for key in ("name", "architecture"):
                val = (e.get(key) or "").strip().lower()
                if val:
                    terms.append(val)
        return terms
    except Exception:
        return []


def _save_failed_architectures(new_failures: list[dict]) -> None:
    """Merge new failures into the persistent failed-architectures checkpoint."""
    existing: dict[str, dict] = {}
    if checkpoint_exists(config.CKPT_FAILED_ARCHITECTURES):
        try:
            data = load_checkpoint(config.CKPT_FAILED_ARCHITECTURES)
            for e in data.get("failed", []):
                k = _norm_arch(e.get("name") or "") or _norm_arch(e.get("architecture") or "")
                if k:
                    existing[k] = e
        except Exception:
            pass

    for e in new_failures:
        k = _norm_arch(e.get("name") or "") or _norm_arch(e.get("architecture") or "")
        if k:
            existing[k] = e

    if existing:
        save_checkpoint(config.CKPT_FAILED_ARCHITECTURES, {"failed": list(existing.values())})


def _run_candidate_smoke(script: str, name: str, index: int) -> dict:
    """Smoke-run a candidate script; return result dict."""
    logger.info("Smoke-running candidate %d: %s", index, name)
    result = code_runner.run_script(script, timeout=config.TIMEOUT_SECONDS, env={"AOI_RANDOM_SEED": "42"}, debug_mode=True)
    smoke = build_smoke_diagnostics(result.stdout, result.duration_ms, context=f"phase1 slot {index}")
    return {
        "index": index,
        "name": name,
        "script": script,
        "slot": index,                              # required by select_full_run_slots
        "status": "smoke_pending_full",             # required by select_full_run_slots
        "smoke_result": {
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_ms": round(result.duration_ms, 1),
            "stderr_tail": result.stderr[-1000:],
        },
        "smoke_metrics": smoke.get("metrics"),
        "smoke_score": smoke.get("score"),
        "smoke_diagnostics": smoke,
        "pruned": smoke.get("pruned", False),
        "prune_reason": smoke.get("prune_reason"),
        "failed": result.returncode != 0 or result.timed_out,
    }


def _run_candidate_full(script: str, name: str, index: int) -> dict:
    """Full training run on one candidate; return metrics dict or None."""
    logger.info("Full-running candidate %d: %s", index, name)
    result = code_runner.run_script(script, timeout=config.TIMEOUT_SECONDS, env={"AOI_RANDOM_SEED": "42"})
    parsed = parse_metrics(result.stdout)
    guarded = metric_guard.guard_metrics(parsed, result.duration_ms, context=f"phase1 full slot {index}") if parsed else None
    return {
        "index": index,
        "name": name,
        "script": script,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_ms": round(result.duration_ms, 1),
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-1000:],
        "metrics": metrics_to_dict(guarded) if guarded else None,
        "status": "success" if (result.returncode == 0 and guarded is not None) else "failed",
    }


def _evaluate_candidates(
    candidate_entries: list[dict],
    modality: str,
) -> list[dict]:
    """Run smoke + selected full runs for all candidates.

    Returns list of scored result dicts sorted best-first.
    Also saves failed architectures to the permanent ban list.
    """
    smoke_results: list[dict] = []
    for entry in candidate_entries:
        smoke = _run_candidate_smoke(entry["script"], entry["name"], entry["index"])
        smoke_results.append({**entry, **smoke})

    # Select which candidates get a full run based on smoke ranking
    slot_map = select_full_run_slots(smoke_results)
    scored: list[dict] = []
    new_failures: list[dict] = []

    for r in smoke_results:
        idx = r["index"]
        name = r["name"]
        arch = r.get("architecture", "")

        if r["failed"] or r["pruned"]:
            result = {**r, "status": "failed" if r["failed"] else "smoke_pruned", "metrics": None}
            new_failures.append({"name": name, "architecture": arch})
            scored.append(result)
            continue

        if idx in slot_map:
            full = _run_candidate_full(r["script"], name, idx)
            result = {**r, **full}
        else:
            result = {**r, "status": "smoke_deferred", "metrics": r["smoke_metrics"]}

        scored.append(result)

    # Persist failed architectures
    if new_failures:
        _save_failed_architectures(new_failures)

    # Sort best-first using acceptance scoring
    def _sort_key(r: dict) -> tuple:
        m = r.get("metrics") or r.get("smoke_metrics")
        if m is None:
            return (1, 1.0, 0.0, 1.0, 0.0)
        from mle_star_agent.shared.acceptance_scoring import acceptance_distance, passes_relaxed_acceptance
        dist = acceptance_distance(m)
        passed = 0 if passes_relaxed_acceptance(m) else 1
        return (passed, dist, -m.get("ng_recall", 0.0), m.get("miss_rate", 1.0), m.get("overkill_rate", 1.0))

    scored.sort(key=_sort_key)
    return scored


# ---------------------------------------------------------------------------
# §3.1.6 — Merger (iterative LLM-driven code-merge)
# ---------------------------------------------------------------------------

def _merge_scripts(base_script: str, other_script: str, base_name: str, other_name: str, token_state: dict) -> Optional[str]:
    """Ask the LLM to incorporate the best architectural ideas from other_script into base_script."""
    system_prompt = (
        "You are an expert PyTorch engineer. You are given two AOI training scripts. "
        "Your task is to produce a SINGLE merged script that incorporates the strongest "
        "architectural ideas from both, while keeping the better-performing base script "
        "as the starting point. The merged script must remain self-contained and follow "
        "ALL mandatory requirements (dry-run support, 9-channel stereo input, weighted loss, "
        "LR schedule with scheduler.step(), threshold sweep, isotonic calibration, and all "
        "required output markers: PROBE_METRICS, EPOCH_LOG, METRICS, CALIBRATION_STATS, "
        "THRESHOLD_CURVE, PREDICTIONS).\n\n"
        "Focus on architectural improvements: different backbone capacity, feature fusion "
        "strategy, loss function variant, or regularization. Do NOT simply average predictions "
        "of two separate models (that would require two training runs). Produce one coherent "
        "training script.\n\n"
        "Return JSON: {\"script\": \"...\", \"architecture\": \"merged\"}"
    )
    user_prompt = (
        f"Base script ({base_name}):\n```python\n{base_script[:6000]}\n```\n\n"
        f"Other script ({other_name}) — extract the best ideas from this:\n```python\n{other_script[:4000]}\n```\n\n"
        "Produce a single merged script that keeps the base script's structure but incorporates "
        "the most promising architectural ideas from the other script. "
        "Return JSON: {\"script\": \"<full merged script>\", \"architecture\": \"merged\"}"
    )
    try:
        response = call_llm_json(
            build_messages(system_prompt, user_prompt),
            model=config.MODEL_PRO,
            max_tokens=8192,
            temperature=0.2,
            token_state=token_state,
        )
        if isinstance(response, dict):
            script = response.get("script", "")
            if script and len(script) > 200:
                return script
    except Exception as exc:
        logger.warning("Merger LLM call failed: %s", exc)
    return None


def _run_merger(
    scored_candidates: list[dict],
    modality: str,
    token_state: dict,
) -> dict:
    """Run the Phase 1 merger: attempt to improve s_0 by merging with other candidates.

    Returns the best result dict (possibly the original s_0, or a merged variant).
    """
    if not scored_candidates:
        return {}

    # Start with the top candidate as s_0
    s_0 = scored_candidates[0]
    best_metrics = s_0.get("metrics") or s_0.get("smoke_metrics")
    best_script = s_0["script"]
    best_name = s_0["name"]

    # Only proceed with candidates that have valid metrics
    others = [r for r in scored_candidates[1:] if (r.get("metrics") or r.get("smoke_metrics")) and r.get("script")]

    if not others:
        logger.info("Merger: no other candidates with valid metrics; keeping s_0=%s", best_name)
        return s_0

    # attempts_completed counts merges that reached the final comparison step
    # (LLM generated + validated + smoke passed). We only stop early once at
    # least one such attempt has been made and it did not improve.
    attempts_completed = 0
    for other in others:
        other_name = other["name"]
        other_script = other["script"]
        logger.info("Merger: trying to merge %s into %s", other_name, best_name)

        merged_script = _merge_scripts(best_script, other_script, best_name, other_name, token_state)
        if merged_script is None:
            logger.info("Merger: LLM merge failed for %s — skipping", other_name)
            continue

        # Validate the merged script
        val_result = validate_script(merged_script, input_modality=modality)
        if not val_result.valid:
            logger.info("Merger: merged script failed validation: %s", val_result.rejection_reasons)
            continue

        # Smoke-run to see if it improves
        smoke = _run_candidate_smoke(merged_script, f"merged_{best_name}+{other_name}", -1)
        merged_metrics = smoke.get("smoke_metrics")

        if smoke["failed"] or smoke["pruned"] or merged_metrics is None:
            logger.info("Merger: merged script failed/pruned smoke run")
            continue

        # This merge reached the comparison step; count it
        attempts_completed += 1

        if best_metrics is None or is_acceptance_improvement(merged_metrics, best_metrics):
            logger.info("Merger: merged script IMPROVES over %s — adopting", best_name)
            best_script = merged_script
            best_metrics = merged_metrics
            best_name = f"merged_{best_name}+{other_name}"
        else:
            logger.info("Merger: merged script does not improve over %s", best_name)
            # Stop after the first genuine non-improving attempt (at least one reached here)
            logger.info("Merger: stopping after first non-improving evaluated attempt")
            break

    return {
        **s_0,
        "script": best_script,
        "name": best_name,
        "metrics": best_metrics,
        "merged": best_name != s_0["name"],
    }


# ---------------------------------------------------------------------------
# §3.1.7 — Save L0 checkpoint
# ---------------------------------------------------------------------------

def _save_l0(best: dict, candidate_entries: list[dict], scored_candidates: list[dict]) -> None:
    """Write CKPT_L0, CKPT_CANDIDATE_SCRIPTS, CKPT_CANDIDATE_SCORES."""
    metrics = best.get("metrics") or best.get("smoke_metrics") or {}

    l0_data = {
        "best_candidate_name": best.get("name", ""),
        "script": best["script"],
        "current_best_score": float(metrics.get("ng_recall", 0.0)),
        "best_miss_rate": float(metrics.get("miss_rate", 1.0)),
        "best_overkill_rate": float(metrics.get("overkill_rate", 1.0)),
        "best_accuracy": float(metrics.get("accuracy", 0.0)),
        "best_f1": float(metrics.get("f1", 0.0)),
    }
    save_checkpoint(config.CKPT_L0, l0_data)
    logger.info("Saved L0: %s (ng_recall=%.4f)", l0_data["best_candidate_name"], l0_data["current_best_score"])

    scripts_data = {
        "scripts": [
            {"name": e["name"], "script": e["script"], "architecture": e.get("architecture", e["name"])}
            for e in candidate_entries
        ]
    }
    save_checkpoint(config.CKPT_CANDIDATE_SCRIPTS, scripts_data)

    scores_data = {
        "scores": [
            {
                "index": r.get("index", i),
                "name": r.get("name", ""),
                "architecture": r.get("architecture", r.get("name", "")),
                "status": r.get("status", "unknown"),
                "smoke_score": r.get("smoke_score"),
                "smoke_metrics": r.get("smoke_metrics"),
                "metrics": r.get("metrics"),
            }
            for i, r in enumerate(scored_candidates)
        ]
    }
    save_checkpoint(config.CKPT_CANDIDATE_SCORES, scores_data)


# ---------------------------------------------------------------------------
# Main Phase 1 node
# ---------------------------------------------------------------------------

def phase1_init_node(state: AgentState) -> dict:
    """Phase 1: data split + retrieval + baseline generation + evaluation + L0.

    Returns a partial state update dict.
    """
    token_state: dict = {"token_count": state.get("tokens_used", 0) or 0}
    debug_mode: bool = state.get("debug_mode", config.DEBUG_MODE)

    # ------------------------------------------------------------------
    # §3.1.1 — Skip check: if L0 + candidate scores already exist, restore
    # ------------------------------------------------------------------
    if checkpoint_exists(config.CKPT_CANDIDATE_SCORES) and checkpoint_exists(config.CKPT_L0):
        logger.info("Phase 1 skip: CKPT_L0 and CKPT_CANDIDATE_SCORES already exist — restoring state.")
        l0 = load_checkpoint(config.CKPT_L0)
        return {
            "current_phase": "refine",
            "current_best_score": float(l0.get("current_best_score", 0.0)),
            "best_miss_rate": float(l0.get("best_miss_rate", 1.0)),
            "best_overkill_rate": float(l0.get("best_overkill_rate", 1.0)),
            "best_accuracy": float(l0.get("best_accuracy", 0.0)),
            "best_f1": float(l0.get("best_f1", 0.0)),
            "best_candidate_name": l0.get("best_candidate_name", ""),
            "best_pipeline": {"script": l0.get("script", "")},
            "knowledge_base": load_kb_from_disk(),
            "outer_iteration": 0,
            "inner_iteration": 0,
            "no_improve_count": 0,
            "tokens_used": token_state["token_count"],
        }

    # ------------------------------------------------------------------
    # §3.1.2 — Data split
    # ------------------------------------------------------------------
    if checkpoint_exists(config.CKPT_DATA_SPLIT):
        logger.info("Loading existing data split from checkpoint.")
        data_split = load_checkpoint(config.CKPT_DATA_SPLIT)
    else:
        dataset_path = state.get("dataset_path", "")
        if not dataset_path:
            import glob as _glob
            lot_folders = sorted(_glob.glob(str(config.PROJECT_ROOT / config.DATASET_GLOB)))
        else:
            import glob as _glob
            # Keep only subdirectories — a flat dataset (images directly in
            # dataset_path) produces no subdirs, so fall back to the path itself.
            candidates = sorted(_glob.glob(str(Path(dataset_path) / "*")))
            lot_folders = [p for p in candidates if Path(p).is_dir()]
            if not lot_folders:
                lot_folders = [dataset_path]

        logger.info("Building data split from %d lot folder(s): %s", len(lot_folders), lot_folders)
        data_split = build_data_split(lot_folders)
        save_checkpoint(config.CKPT_DATA_SPLIT, data_split)
        logger.info("Data split saved: train=%d val=%d test=%d",
                    len(data_split["train"]), len(data_split["val"]), len(data_split["test"]))

    modality: str = data_split.get("metadata", {}).get("input_modality", "stereo")

    # ------------------------------------------------------------------
    # §3.1.3 — Retriever: web search for M=4 candidate architectures
    # ------------------------------------------------------------------
    if checkpoint_exists(config.CKPT_CANDIDATE_SCRIPTS):
        logger.info("Candidate scripts already exist; loading from checkpoint.")
        scripts_ckpt = load_checkpoint(config.CKPT_CANDIDATE_SCRIPTS)
        candidate_entries = scripts_ckpt.get("scripts", [])
        # Filter out any hard-excluded architectures
        runtime_failed = _load_runtime_failed_arch_terms()
        all_excluded = _HARD_EXCLUDED + [t.lower() for t in runtime_failed]
        candidate_entries = [
            e for e in candidate_entries
            if not any(t in (e.get("name", "") + " " + e.get("architecture", "")).lower() for t in all_excluded)
        ]
        if len(candidate_entries) >= 3:
            logger.info("Loaded %d valid candidate scripts from checkpoint.", len(candidate_entries))
        else:
            logger.info("Only %d valid scripts in checkpoint; running fresh retrieval.", len(candidate_entries))
            candidate_entries = []  # fall through to regenerate
    else:
        candidate_entries = []

    if not candidate_entries:
        # Retrieve candidates via web search + LLM
        if not debug_mode:
            retrieved = _retrieve_candidates(data_split, token_state)
        else:
            # Debug mode: skip web search, use minimal built-in list
            logger.info("DRY_RUN: skipping web search, using built-in candidates")
            retrieved = [
                {"model_name": "EfficientNet-B0", "description": "Lightweight CNN baseline.", "example_code": ""},
                {"model_name": "MobileNetV3-Small", "description": "Very lightweight CNN.", "example_code": ""},
                {"model_name": "ResNet-18", "description": "Classic CNN baseline.", "example_code": ""},
                {"model_name": "DINOv2-ViT-S/14", "description": "Frozen ViT feature extractor.", "example_code": ""},
            ]

        # ------------------------------------------------------------------
        # §3.1.4 — Baseline coder: generate scripts
        # ------------------------------------------------------------------
        candidate_entries = []
        runtime_failed = _load_runtime_failed_arch_terms()
        all_excluded = _HARD_EXCLUDED + [t.lower() for t in runtime_failed]

        for i, candidate in enumerate(retrieved[:_M]):
            name = candidate.get("model_name", f"candidate_{i}")

            # Skip hard-excluded architectures
            if any(t in name.lower() for t in all_excluded):
                logger.info("Skipping hard-excluded architecture: %s", name)
                continue

            if debug_mode:
                # In debug mode, generate a minimal stub script that satisfies the validator
                script = _build_dry_run_stub_script(name, i, modality)
            else:
                script = _generate_script(candidate, modality, token_state)

            if script is None:
                logger.warning("Script generation returned None for: %s", name)
                continue

            # Validate before adding
            val_result = validate_script(script, input_modality=modality)
            if not val_result.valid:
                logger.warning("Script for %s failed validation: %s", name, val_result.rejection_reasons)
                # Still include it — evaluator will record the failure and ban the arch
            candidate_entries.append({
                "index": i,
                "name": name,
                "script": script,
                "architecture": name,
                "validated": val_result.valid,
            })

        if candidate_entries:
            scripts_data = {
                "scripts": [
                    {"name": e["name"], "script": e["script"], "architecture": e["architecture"]}
                    for e in candidate_entries
                ]
            }
            save_checkpoint(config.CKPT_CANDIDATE_SCRIPTS, scripts_data)

    else:
        # Re-attach index for candidates loaded from checkpoint
        candidate_entries = [
            {**e, "index": i, "validated": True}
            for i, e in enumerate(candidate_entries)
        ]

    if not candidate_entries:
        logger.error("Phase 1: no candidate scripts generated — aborting with empty L0.")
        return {"current_phase": "refine", "error": "Phase 1: no candidate scripts generated"}

    # ------------------------------------------------------------------
    # §3.1.5 — Candidate evaluator
    # ------------------------------------------------------------------
    if checkpoint_exists(config.CKPT_CANDIDATE_SCORES):
        logger.info("Candidate scores already exist; loading from checkpoint.")
        scores_ckpt = load_checkpoint(config.CKPT_CANDIDATE_SCORES)
        scored_candidates = scores_ckpt.get("scores", [])
    else:
        scored_candidates = _evaluate_candidates(candidate_entries, modality)

    if not scored_candidates:
        logger.error("Phase 1: candidate evaluation produced no results.")
        return {"current_phase": "refine", "error": "Phase 1: evaluation produced no results"}

    # ------------------------------------------------------------------
    # §3.1.6 — Merger
    # ------------------------------------------------------------------
    # Match scored results back to full scripts from candidate_entries
    name_to_script = {e["name"]: e["script"] for e in candidate_entries}
    for r in scored_candidates:
        if not r.get("script"):
            r["script"] = name_to_script.get(r.get("name", ""), "")

    if not debug_mode:
        best_result = _run_merger(scored_candidates, modality, token_state)
    else:
        # Skip merger in debug mode — just pick the top candidate
        valid_scored = [r for r in scored_candidates if r.get("script")]
        best_result = valid_scored[0] if valid_scored else scored_candidates[0]

    if not best_result or not best_result.get("script"):
        # Fallback to first candidate with a script
        for r in scored_candidates:
            if r.get("script"):
                best_result = r
                break

    # ------------------------------------------------------------------
    # §3.1.7 — Save L0
    # ------------------------------------------------------------------
    _save_l0(best_result, candidate_entries, scored_candidates)

    # Build final metrics
    best_metrics = best_result.get("metrics") or best_result.get("smoke_metrics") or {}

    return {
        "current_phase": "refine",
        "data_split": data_split,
        "candidate_scripts": [e["script"] for e in candidate_entries],
        "candidate_scores": [
            {
                "name": r.get("name"),
                "status": r.get("status"),
                "metrics": r.get("metrics") or r.get("smoke_metrics"),
            }
            for r in scored_candidates
        ],
        "best_pipeline": {"script": best_result.get("script", "")},
        "best_candidate_name": best_result.get("name", ""),
        "current_best_score": float(best_metrics.get("ng_recall", 0.0)),
        "best_miss_rate": float(best_metrics.get("miss_rate", 1.0)),
        "best_overkill_rate": float(best_metrics.get("overkill_rate", 1.0)),
        "best_accuracy": float(best_metrics.get("accuracy", 0.0)),
        "best_f1": float(best_metrics.get("f1", 0.0)),
        "knowledge_base": load_kb_from_disk(),
        "outer_iteration": 0,
        "inner_iteration": 0,
        "no_improve_count": 0,
        "tokens_used": token_state["token_count"],
    }


# ---------------------------------------------------------------------------
# DRY_RUN stub script builder (used in debug mode to skip LLM generation)
# ---------------------------------------------------------------------------

# Template string with __PLACEHOLDERS__ for the few context-dependent values.
# Using str.replace() substitution instead of an f-string avoids double-escaping
# the many {variable} references inside the generated script's own print statements.
_STUB_BODY = """\
import json, os, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

DRY_RUN         = os.getenv("DRY_RUN") == "1"
DRY_RUN_EPOCHS  = int(os.getenv("DRY_RUN_EPOCHS", "1"))
DRY_RUN_SAMPLES = int(os.getenv("DRY_RUN_SAMPLES", "10"))

DATA_SPLIT_PATH = "__DATA_SPLIT_PATH__"
with open(DATA_SPLIT_PATH) as f:
    data_split = json.load(f)

train_samples = data_split["train"]
val_samples   = data_split["val"]
test_samples  = data_split["test"]

if DRY_RUN:
    train_samples = train_samples[:DRY_RUN_SAMPLES]
    val_samples   = val_samples[:DRY_RUN_SAMPLES]
    test_samples  = test_samples[:DRY_RUN_SAMPLES]

epochs = DRY_RUN_EPOCHS if DRY_RUN else 20
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

n_ng = max(1, sum(1 for s in train_samples if s["label"] == "NG"))
n_g  = max(1, sum(1 for s in train_samples if s["label"] == "G"))
pos_weight = torch.tensor([n_g / n_ng])


# ----- DATA LOADING -----
def _load_one(sample):
# data_loading
# augmentation
__IMAGE_LOAD_BLOCK__
    return img_tensor, 1.0 if sample["label"] == "NG" else 0.0


def _make_tensors(samples):
    xs, ys = [], []
    for s in samples:
        try:
            x, y = _load_one(s)
            xs.append(x)
            ys.append(torch.tensor(y, dtype=torch.float32))
        except Exception:
            pass
    if not xs:
        return torch.zeros(1, __IN_CHANNELS__, 64, 64), torch.zeros(1)
    return torch.stack(xs), torch.stack(ys)


# ----- MODEL ARCHITECTURE -----
# model_architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(__IN_CHANNELS__, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


model = SimpleModel().to(device)

# ----- LOSS FUNCTION -----
# loss_function
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

# ----- PROBE -----
model.eval()
probe_x, probe_y = _make_tensors(val_samples[:5])
with torch.no_grad():
    probe_logits = model(probe_x.to(device))
    probe_probs = torch.sigmoid(probe_logits).cpu().numpy()
probe_y_np = probe_y.numpy()
pg  = float(probe_probs[probe_y_np == 0].mean()) if (probe_y_np == 0).any() else 0.5
png = float(probe_probs[probe_y_np == 1].mean()) if (probe_y_np == 1).any() else 0.5
print("PROBE_METRICS: " + json.dumps({
    "ng_recall": 0.5, "overkill_rate": 0.5,
    "G_prob_mean": round(pg, 4), "NG_prob_mean": round(png, 4),
    "should_continue": True, "reason": "stub probe",
}))

# ----- TRAIN -----
train_x, train_y = _make_tensors(train_samples)
val_x,   val_y   = _make_tensors(val_samples)
test_x,  test_y  = _make_tensors(test_samples)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(train_x.to(device)), train_y.to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        val_logits_ep = model(val_x.to(device))
        val_loss_ep   = criterion(val_logits_ep, val_y.to(device)).item()
        vprobs = torch.sigmoid(val_logits_ep).cpu().numpy()
        vy_np  = val_y.numpy()
    vng = float((vprobs[vy_np == 1] >= 0.5).mean()) if (vy_np == 1).any() else 0.0
    vok = float((vprobs[vy_np == 0] >= 0.5).mean()) if (vy_np == 0).any() else 0.0
    print("EPOCH_LOG: " + json.dumps({
        "epoch": epoch, "train_loss": round(float(loss.item()), 4),
        "val_loss": round(val_loss_ep, 4),
        "val_ng_recall": round(vng, 4), "val_overkill": round(vok, 4),
    }))

# ----- CALIBRATION -----
# calibration
model.eval()
with torch.no_grad():
    val_raw_logits = model(val_x.to(device)).cpu().numpy()
val_y_np = val_y.numpy()
val_probs_raw = 1.0 / (1.0 + np.exp(-val_raw_logits))
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(val_probs_raw, val_y_np)
val_probs_cal = iso.transform(val_probs_raw)
gm  = val_y_np == 0
ngm = val_y_np == 1
print("CALIBRATION_STATS: " + json.dumps({
    "G_prob_mean":  round(float(val_probs_cal[gm].mean())  if gm.any()  else 0.0, 4),
    "G_prob_std":   round(float(val_probs_cal[gm].std())   if gm.any()  else 0.0, 4),
    "NG_prob_mean": round(float(val_probs_cal[ngm].mean()) if ngm.any() else 0.5, 4),
    "NG_prob_std":  round(float(val_probs_cal[ngm].std())  if ngm.any() else 0.0, 4),
}))

# ----- THRESHOLD SWEEP -----
# threshold_selection
curve = []
for ti in range(1, 100):
    t = round(ti / 100.0, 2)
    pred_v = (val_probs_cal >= t).astype(int)
    _tp = int(((pred_v == 1) & (val_y_np == 1)).sum())
    _tn = int(((pred_v == 0) & (val_y_np == 0)).sum())
    _fp = int(((pred_v == 1) & (val_y_np == 0)).sum())
    _fn = int(((pred_v == 0) & (val_y_np == 1)).sum())
    _r  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 1.0
    _ok = _fp / (_tn + _fp) if (_tn + _fp) > 0 else 0.0
    _mr = _fn / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
    _ac = (_tp + _tn) / max(1, _tp + _tn + _fp + _fn)
    curve.append({"t": t, "recall": round(_r, 4), "overkill": round(_ok, 4),
                  "miss_rate": round(_mr, 4), "accuracy": round(_ac, 4)})
print("THRESHOLD_CURVE: " + json.dumps(curve))

# ----- SELECT THRESHOLD -----
all_cands = []
for item in curve:
    p_v = (val_probs_cal >= item["t"]).astype(int)
    fp_c = int(((p_v == 1) & (val_y_np == 0)).sum())
    all_cands.append((item["t"], item["miss_rate"], item["overkill"], fp_c))
survivors = [c for c in all_cands if c[3] <= 2]
if survivors:
    min_miss = min(c[1] for c in survivors)
    stage1   = [c for c in survivors if c[1] == min_miss]
    best_t   = min(stage1, key=lambda c: c[2])[0]
elif all_cands:
    best_t = min(all_cands, key=lambda c: (c[3], c[1]))[0]
else:
    best_t = 0.5

# ----- TEST METRICS -----
model.eval()
with torch.no_grad():
    test_logits = model(test_x.to(device)).cpu().numpy()
test_probs_raw = 1.0 / (1.0 + np.exp(-test_logits))
test_probs_cal = iso.transform(test_probs_raw)
test_y_np = test_y.numpy()
pred_test = (test_probs_cal >= best_t).astype(int)
tp = int(((pred_test == 1) & (test_y_np == 1)).sum())
tn = int(((pred_test == 0) & (test_y_np == 0)).sum())
fp = int(((pred_test == 1) & (test_y_np == 0)).sum())
fn = int(((pred_test == 0) & (test_y_np == 1)).sum())
total = tp + tn + fp + fn
acc    = (tp + tn) / max(1, total)
recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
mr     = fn / (tp + fn) if (tp + fn) > 0 else 0.0
ok     = fp / (tn + fp) if (tn + fp) > 0 else 0.0
prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
try:
    roc = float(roc_auc_score(test_y_np, test_probs_cal)) if len(set(test_y_np.tolist())) > 1 else 0.0
except Exception:
    roc = 0.0
ng_arr = test_probs_cal[test_y_np == 1]
g_arr  = test_probs_cal[test_y_np == 0]
prob_gap = float(ng_arr.mean() - g_arr.mean()) if (len(ng_arr) > 0 and len(g_arr) > 0) else 0.0
print("METRICS: " + json.dumps({
    "accuracy": round(acc, 4), "ng_recall": round(recall, 4),
    "miss_rate": round(mr, 4), "overkill_rate": round(ok, 4),
    "f1": round(f1, 4), "avg_latency_ms": 1.0, "threshold": best_t,
    "ng_count": int(test_y_np.sum()), "g_count": int((1 - test_y_np).sum()),
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    "roc_auc": round(roc, 4), "prob_gap": round(prob_gap, 4),
}))

# ----- PREDICTIONS -----
predictions = []
for _i, _s in enumerate(test_samples):
    _p = float(test_probs_cal[_i]) if _i < len(test_probs_cal) else 0.5
    predictions.append({
        "sample_id": _s.get("sample_id", str(_i)),
        "true_label": _s["label"],
        "predicted_label": "NG" if _p >= best_t else "G",
        "ng_probability": round(_p, 4),
        "threshold": best_t,
    })
print("PREDICTIONS: " + json.dumps(predictions))
"""


def _build_dry_run_stub_script(name: str, index: int, modality: str) -> str:
    """Return a minimal but validator-compliant stub script for dry-run/testing."""
    stereo_load = modality == "stereo"
    in_channels = 9 if stereo_load else 3

    if stereo_load:
        image_load_block = (
            '    img_l = Image.open(sample["img_l"]).convert("RGB").resize((64, 64))\n'
            '    img_r = Image.open(sample["img_r"]).convert("RGB").resize((64, 64))\n'
            "    tform = transforms.ToTensor()\n"
            "    il, ir = tform(img_l), tform(img_r)\n"
            "    img_tensor = torch.cat([il, ir, torch.abs(il - ir)], dim=0)"
        )
    else:
        image_load_block = (
            '    img = Image.open(sample["img"]).convert("RGB").resize((64, 64))\n'
            "    img_tensor = transforms.ToTensor()(img)"
        )

    header = f'#!/usr/bin/env python3\n# Stub training script for {name} (index={index}) — dry-run/test\n'
    body = (
        _STUB_BODY
        .replace("__DATA_SPLIT_PATH__", _DATA_SPLIT_PATH)
        .replace("__IN_CHANNELS__", str(in_channels))
        .replace("__IMAGE_LOAD_BLOCK__", image_load_block)
    )
    return header + body
