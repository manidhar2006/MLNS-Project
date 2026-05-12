"""
amr_hgat — Matched Ablation Package for HGAT TB AMR Prediction.

Arms:
  A  SNP-only baseline (matched protocol)
  B  Pathway-minimal  (hub-pruned, +mtu00074, no gene hub, layer-2 reverse masked)
  C  Pathway + KO     (adds KO nodes bridging pathway-orphan genes)
  D  Pathway + KO + Drug  (adds drug nodes + per-drug masked pathway attention)
"""
from .data import load_data, DRUG_COLS_PRIMARY, DRUG_COLS_SECONDARY, DRUG_COLS_ALL
from .graph_builders import build_graph
from .model import HGATModel
from .train import cross_validate
from .metrics import compute_metrics, find_best_thresholds
from .splits import make_folds
from .attention import run_attention_analysis, SECOND_LINE_DRUGS, KNOWN_RESISTANCE_GENES

__all__ = [
    "load_data",
    "DRUG_COLS_PRIMARY",
    "DRUG_COLS_SECONDARY",
    "DRUG_COLS_ALL",
    "build_graph",
    "HGATModel",
    "cross_validate",
    "compute_metrics",
    "find_best_thresholds",
    "make_folds",
    "run_attention_analysis",
    "SECOND_LINE_DRUGS",
    "KNOWN_RESISTANCE_GENES",
]
