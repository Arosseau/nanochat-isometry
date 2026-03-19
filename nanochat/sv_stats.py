"""
Singular value statistics for monitoring isometry during training.

Computes per-matrix stats (max/min/mean squared singular value, condition number,
effective rank) for all 2D weight matrices in transformer.h, then aggregates to:
  - Global scalars: mean across all matrices (5 metrics)
  - Per-type scalars: mean per weight-type group (7 types × 5 metrics = 35 scalars)
  - Histograms: distribution of cond/eff_rank/max_sv2 across all matrices

Logged to wandb (~40 scalars + 3 histograms per eval) and saved locally as JSONL
for offline matplotlib analysis.

JSONL format (one line per eval step):
  {"step": N, "global": {...}, "by_type": {...}, "per_matrix": [...]}

Loading for matplotlib:
  import json, pandas as pd
  records = [json.loads(l) for l in open("sv_stats.jsonl")]
  # Global metrics over time:
  df = pd.DataFrame([{"step": r["step"], **r["global"]} for r in records])
  # Per-matrix data for layer heatmaps at a given step:
  pm = pd.DataFrame(records[i]["per_matrix"])  # columns: layer, type, cond, eff_rank, ...

Effective rank definition: Roy & Vetterli (2007), IEEE ICASSP.
  eff_rank = exp(H(σ²/Σσ²))  where H is Shannon entropy.
  eff_rank = 1 means rank-1 (degenerate), eff_rank = min(m,n) means perfectly flat spectrum.

Reference: "Preserving Plasticity in Continual Learning via Dynamical Isometry"
"""

import math
import json
import os
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# Weight types present in nanochat transformer blocks
_WEIGHT_TYPES = ('c_q', 'c_k', 'c_v', 'c_proj_attn', 've_gate', 'c_fc', 'c_proj_mlp')


def _parse_matrix_name(name: str) -> tuple[int, str]:
    """
    Parse a parameter name to (layer_idx, weight_type).

    Handles names like:
      'transformer.h.3.attn.c_q.weight'   → (3, 'c_q')
      'transformer.h.3.attn.c_proj.weight' → (3, 'c_proj_attn')
      'transformer.h.3.attn.ve_gate.weight'→ (3, 've_gate')
      'transformer.h.3.mlp.c_fc.weight'    → (3, 'c_fc')
      'transformer.h.3.mlp.c_proj.weight'  → (3, 'c_proj_mlp')
    """
    parts = name.split('.')
    layer_idx = int(parts[2])
    subpath = '.'.join(parts[3:])

    if 'attn.c_q' in subpath:
        wtype = 'c_q'
    elif 'attn.c_k' in subpath:
        wtype = 'c_k'
    elif 'attn.c_v' in subpath:
        wtype = 'c_v'
    elif 'attn.c_proj' in subpath:
        wtype = 'c_proj_attn'
    elif 'attn.ve_gate' in subpath:
        wtype = 've_gate'
    elif 'mlp.c_fc' in subpath:
        wtype = 'c_fc'
    elif 'mlp.c_proj' in subpath:
        wtype = 'c_proj_mlp'
    else:
        wtype = 'other'
    return layer_idx, wtype


def _effective_rank(sv2: Tensor) -> float:
    """
    Effective rank (Roy & Vetterli 2007): exp(H(σ²/Σσ²)).

    - eff_rank ≈ 1    → degenerate (rank-1 matrix)
    - eff_rank = k    → k singular values carry all the energy equally
    - eff_rank = min(m,n) → perfectly flat spectrum (isometric case)
    """
    sv2 = sv2.float()
    total = sv2.sum()
    if total < 1e-10:
        return 0.0
    p = sv2 / total
    mask = p > 1e-10
    entropy = -(p[mask] * p[mask].log()).sum().item()
    return math.exp(entropy)


_NAN_STATS = {'max_sv2': float('nan'), 'min_sv2': float('nan'),
              'mean_sv2': float('nan'), 'cond': float('nan'), 'eff_rank': float('nan')}


def _matrix_stats(param: Tensor) -> dict:
    """Compute all stats for a single weight matrix. param should be 2D."""
    W = param.float()  # fp32 for numerical stability
    try:
        sv = torch.linalg.svdvals(W)  # singular values only — faster than full SVD
    except Exception:
        return dict(_NAN_STATS)
    if not torch.isfinite(sv).all():
        return dict(_NAN_STATS)
    sv2 = sv ** 2
    min_sv = sv.min().clamp(min=1e-10)
    return {
        'max_sv2': sv2.max().item(),
        'min_sv2': sv2.min().item(),
        'mean_sv2': sv2.mean().item(),
        'cond': (sv.max() / min_sv).item(),  # σ_max / σ_min
        'eff_rank': _effective_rank(sv2),
    }


def _avg_stats(records: list[dict]) -> dict:
    """Average each stat key across a list of per-matrix stat dicts, skipping NaN entries."""
    keys = ('max_sv2', 'min_sv2', 'mean_sv2', 'cond', 'eff_rank')
    result = {}
    for k in keys:
        vals = [r[k] for r in records if math.isfinite(r[k])]
        result[k] = sum(vals) / len(vals) if vals else float('nan')
    return result


@torch.no_grad()
def compute_sv_stats(model: nn.Module) -> dict:
    """
    Compute singular value statistics for all 2D weight matrices in transformer.h.

    Should be called on orig_model (uncompiled), not the torch.compile'd model.
    Automatically casts weights to fp32 for numerical stability.
    Uses torch.linalg.svdvals (singular values only, no vectors) for efficiency.

    Returns a dict with:
      'per_matrix': list of per-matrix stat dicts (layer, type, shape, max_sv2, ...)
      'by_type':    dict[type_name → averaged stats dict]
      'global':     dict of averages across all matrices
      'wandb':      flat dict ready for wandb.log() (sv/global/*, sv/{type}/*)
      'histograms': dict[metric → list of floats] for wandb.Histogram
    """
    per_matrix = []
    for name, param in model.named_parameters():
        if param.ndim != 2 or 'transformer.h.' not in name:
            continue
        layer_idx, wtype = _parse_matrix_name(name)
        stats = _matrix_stats(param)
        per_matrix.append({
            'name': name,
            'layer': layer_idx,
            'type': wtype,
            'shape': list(param.shape),
            **stats,
        })

    if not per_matrix:
        return {'per_matrix': [], 'by_type': {}, 'global': {}, 'wandb': {}, 'histograms': {}}

    # Per-type aggregation
    type_groups = defaultdict(list)
    for m in per_matrix:
        type_groups[m['type']].append(m)
    by_type = {t: _avg_stats(records) for t, records in type_groups.items()}

    # Global aggregation (mean across all matrices regardless of type)
    global_stats = _avg_stats(per_matrix)

    # Flat wandb dict: sv/global/* and sv/{type}/*
    wandb_dict = {f'sv/global/{k}': v for k, v in global_stats.items()}
    for wtype, stats in by_type.items():
        for k, v in stats.items():
            wandb_dict[f'sv/{wtype}/{k}'] = v

    # Histograms: distribution of key metrics across all matrices (NaN-filtered)
    histograms = {
        'cond':     [m['cond']     for m in per_matrix if math.isfinite(m['cond'])],
        'eff_rank': [m['eff_rank'] for m in per_matrix if math.isfinite(m['eff_rank'])],
        'max_sv2':  [m['max_sv2']  for m in per_matrix if math.isfinite(m['max_sv2'])],
    }

    return {
        'per_matrix': per_matrix,
        'by_type':    by_type,
        'global':     global_stats,
        'wandb':      wandb_dict,
        'histograms': histograms,
    }


def save_sv_stats(stats: dict, step: int, filepath: str, flops: float = 0.0) -> None:
    """
    Append a JSONL record to filepath (one line per eval step).

    Record format:
      {"step": N, "flops": F, "global": {...}, "by_type": {...}, "per_matrix": [...]}

    'per_matrix' entries include 'layer' and 'type' fields, enabling:
      - Layer-by-layer heatmaps: pivot on (layer, type)
      - Per-type evolution: group by type, plot over step
      - Full distribution: use all cond/eff_rank values at a given step

    Example (matplotlib):
      records = [json.loads(l) for l in open("sv_stats.jsonl")]
      steps = [r["step"] for r in records]
      cond  = [r["global"]["cond"] for r in records]
      plt.plot(steps, cond, label="mean condition number")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    record = {
        'step':       step,
        'flops':      flops,
        'global':     stats['global'],
        'by_type':    stats['by_type'],
        'per_matrix': stats['per_matrix'],
    }
    with open(filepath, 'a') as f:
        f.write(json.dumps(record) + '\n')
