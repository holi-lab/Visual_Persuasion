#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze reasoning faithfulness and correctness-aware reasoning quality.

Usage:
  python analyze_reasoning_quality.py <original_dir> <perturbed_dir> [control_dir] \
      --metric margin \
      --csv items.csv \
      --summary-csv summary.csv

Arguments:
  original_dir  : full/original-context predictions, one .jsonl per method/model
  perturbed_dir : predictions after removing the reasoning segment judged important
  control_dir   : optional control/unrelated perturbation predictions

Core metrics:
  1) Faithfulness / decision relevance
     By default, this is margin-based RSD:

       MarginRSD =
         [ margin_orig(y_support) - margin_pert(y_support) ]

       where

         margin(y_support) = logit(y_support) - logit(y_opposite)

     If control_dir is provided, the primary faithfulness score is adjusted:

       Adjusted MarginRSD =
         [margin_orig - margin_pert] - [margin_orig - margin_ctrl]
         = margin_ctrl - margin_pert

     Larger values mean the removed reasoning segment was more decision-relevant.

  2) Correctness-aware quality proxy

       CorrectSupportRSD =
         max(primary_RSD, 0), if original prediction is correct
         0,                   if original prediction is wrong
         None,                if no gold answer is available

     This prevents "faithful but wrong" reasoning from receiving a high quality score.

Expected JSONL fields:
  Required:
    id
    yes_logits / yes_logit
    no_logits  / no_logit

  Optional:
    pred
    answer / label / gold / gold_answer / target

Notes:
  - The default metric is margin-based, not raw-logit-based.
  - Raw logit RSD is still saved in the per-item CSV for inspection.
  - This script does NOT do method-vs-method pairwise ranking.
  - It reports thresholded Correct-Support Rates P(q>tau), default tau in {0, 0.25, 0.5, 0.75}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Label = str
Row = Dict[str, Any]


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def normalize_label(x: Any) -> Optional[Label]:
    """Normalize a free-form Yes/No-like value to 'Yes' or 'No'."""
    if x is None:
        return None

    s = str(x).strip()
    if not s:
        return None

    sl = s.lower()

    # Exact / common variants
    if sl in {"yes", "y", "true", "1", "correct", "positive"}:
        return "Yes"
    if sl in {"no", "n", "false", "0", "incorrect", "negative"}:
        return "No"

    # Conservative parsing for longer model outputs.
    # Do not parse "not yes" as Yes.
    prefix = sl[:32].strip()
    if prefix.startswith("yes"):
        return "Yes"
    if prefix.startswith("no"):
        return "No"

    return None


def opposite(label: Label) -> Label:
    if label == "Yes":
        return "No"
    if label == "No":
        return "Yes"
    raise ValueError(f"Unknown binary label: {label}")


def logit(row: Row, label: Label) -> float:
    if label == "Yes":
        return float(row["yes_logits"])
    if label == "No":
        return float(row["no_logits"])
    raise ValueError(f"Unknown binary label: {label}")


def support_margin(row: Row, support_label: Label) -> float:
    """Binary support margin: logit(support) - logit(opposite)."""
    return logit(row, support_label) - logit(row, opposite(support_label))


def get_gold(row: Row) -> Optional[Label]:
    """Read gold label from common field names."""
    for key in ["answer", "label", "gold", "gold_answer", "target", "gold_label"]:
        if key in row:
            y = normalize_label(row.get(key))
            if y is not None:
                return y
    return None


def get_pred(row: Row) -> Optional[Label]:
    """
    Read prediction from row['pred'] if possible.
    If pred is unavailable or not parseable, fall back to Yes/No logits.
    """
    pred = normalize_label(row.get("pred"))
    if pred is not None:
        return pred

    # Sometimes the model output itself contains the decision.
    pred = normalize_label(row.get("output"))
    if pred is not None:
        return pred

    if "yes_logits" in row and "no_logits" in row:
        return "Yes" if float(row["yes_logits"]) >= float(row["no_logits"]) else "No"

    return None


def get_support_label(original_row: Row) -> Tuple[Optional[Label], Optional[bool], Optional[Label], Optional[Label]]:
    """
    Return:
      y_support:
        gold answer if original prediction is correct;
        original prediction if original prediction is wrong;
        original prediction if gold is unavailable.
      is_correct:
        True/False if gold is available, otherwise None.
      pred:
        original prediction.
      gold:
        gold answer if available.
    """
    pred = get_pred(original_row)
    if pred is None:
        return None, None, None, get_gold(original_row)

    gold = get_gold(original_row)
    if gold is None:
        return pred, None, pred, None

    is_correct = (pred == gold)
    if is_correct:
        return gold, True, pred, gold
    return pred, False, pred, gold


def sort_key_id(x: str) -> Tuple[int, Any]:
    """Sort numeric IDs numerically and other IDs lexicographically."""
    try:
        return (0, int(x))
    except Exception:
        return (1, x)


def finite_values(xs: Iterable[Optional[float]]) -> List[float]:
    out = []
    for x in xs:
        if x is None:
            continue
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            continue
        out.append(float(x))
    return out


def mean(xs: Sequence[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def median(xs: Sequence[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def stdev(xs: Sequence[float]) -> Optional[float]:
    return statistics.stdev(xs) if len(xs) > 1 else None


def p_positive(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(1 for x in xs if x > 0) / len(xs)


def p_greater_than(xs: Sequence[float], threshold: float) -> Optional[float]:
    """Return P(x > threshold). Uses a strict threshold, matching P(q>0)."""
    if not xs:
        return None
    return sum(1 for x in xs if x > threshold) / len(xs)


def n_greater_than(xs: Sequence[float], threshold: float) -> int:
    return sum(1 for x in xs if x > threshold)


def threshold_key(prefix: str, threshold: float) -> str:
    """Stable CSV-friendly key for thresholded metrics."""
    s = f"{threshold:g}".replace("-", "neg").replace(".", "p")
    return f"{prefix}_gt_{s}"


def parse_thresholds(s: str) -> List[float]:
    """Parse comma-separated thresholds such as '0,0.25,0.5,0.75'."""
    vals: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    # Keep user order but remove duplicates.
    out: List[float] = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def fmt_threshold(t: float) -> str:
    return f"{t:g}"


def fmt(x: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if x is None:
        return "N/A"
    if signed:
        return f"{x:+.{digits}f}"
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_jsonl(path: str) -> Dict[str, Row]:
    rows: Dict[str, Row] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in r:
                continue

            if "id" not in r:
                continue

            # Normalize old/new logit field names.
            if "yes_logits" not in r and "yes_logit" in r:
                r["yes_logits"] = r["yes_logit"]
            if "no_logits" not in r and "no_logit" in r:
                r["no_logits"] = r["no_logit"]

            if "yes_logits" not in r or "no_logits" not in r:
                continue

            try:
                r["yes_logits"] = float(r["yes_logits"])
                r["no_logits"] = float(r["no_logits"])
            except Exception:
                continue

            rows[str(r["id"])] = r

    return rows


def jsonl_files(directory: str) -> List[str]:
    return sorted(f for f in os.listdir(directory) if f.endswith(".jsonl"))


# ---------------------------------------------------------------------
# Item-level metrics
# ---------------------------------------------------------------------

def compute_item_metrics(
    model: str,
    id_: str,
    original_row: Row,
    perturbed_row: Row,
    control_row: Optional[Row],
    metric: str = "margin",
) -> Optional[Dict[str, Any]]:
    """
    Compute raw and adjusted RSD for one example.
    metric controls only which score becomes primary_rsd.
    Both logit-based and margin-based scores are always computed.
    """
    y_support, is_correct, pred, gold = get_support_label(original_row)
    if y_support is None or pred is None:
        return None

    # Perturbed prediction and flip.
    pert_pred = get_pred(perturbed_row)
    flipped = None if pert_pred is None else (pred != pert_pred)

    # Confidence of the original decision.
    orig_conf = abs(float(original_row["yes_logits"]) - float(original_row["no_logits"]))

    # Raw logit RSD:
    #   logit_orig(y_support) - logit_pert(y_support)
    logit_rsd_raw = logit(original_row, y_support) - logit(perturbed_row, y_support)

    # Raw margin RSD:
    #   margin_orig(y_support) - margin_pert(y_support)
    margin_rsd_raw = support_margin(original_row, y_support) - support_margin(perturbed_row, y_support)

    logit_rsd_adjusted = None
    margin_rsd_adjusted = None

    if control_row is not None:
        # Adjusted = (orig - pert) - (orig - ctrl) = ctrl - pert.
        logit_rsd_adjusted = logit(control_row, y_support) - logit(perturbed_row, y_support)
        margin_rsd_adjusted = support_margin(control_row, y_support) - support_margin(perturbed_row, y_support)

    if metric == "margin":
        primary_rsd = margin_rsd_adjusted if control_row is not None else margin_rsd_raw
    elif metric == "logit":
        primary_rsd = logit_rsd_adjusted if control_row is not None else logit_rsd_raw
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Correctness-aware quality proxy.
    # If gold is unavailable, quality is unknown rather than zero.
    if is_correct is None:
        correct_support_rsd = None
    elif is_correct:
        correct_support_rsd = max(primary_rsd, 0.0)
    else:
        correct_support_rsd = 0.0

    return {
        "model": model,
        "id": id_,
        "image": original_row.get("image", ""),
        "reason": original_row.get("reason", ""),
        "output": original_row.get("output", original_row.get("pred", "")),
        "pred": pred,
        "gold": gold if gold is not None else "",
        "is_correct": is_correct,
        "perturbed_pred": pert_pred if pert_pred is not None else "",
        "flipped": flipped,
        "support_label": y_support,
        "orig_yes_logit": float(original_row["yes_logits"]),
        "orig_no_logit": float(original_row["no_logits"]),
        "pert_yes_logit": float(perturbed_row["yes_logits"]),
        "pert_no_logit": float(perturbed_row["no_logits"]),
        "orig_confidence_abs_margin": orig_conf,
        "logit_rsd_raw": logit_rsd_raw,
        "logit_rsd_adjusted": logit_rsd_adjusted,
        "margin_rsd_raw": margin_rsd_raw,
        "margin_rsd_adjusted": margin_rsd_adjusted,
        "primary_rsd": primary_rsd,
        "faithfulness": primary_rsd,
        "correct_support_rsd": correct_support_rsd,
    }


def collect_items_for_model(
    model: str,
    original_rows: Dict[str, Row],
    perturbed_rows: Dict[str, Row],
    control_rows: Optional[Dict[str, Row]],
    metric: str,
    forced_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    ids = set(original_rows) & set(perturbed_rows)
    if control_rows is not None:
        ids &= set(control_rows)
    if forced_ids is not None:
        ids &= forced_ids

    items = []
    for id_ in sorted(ids, key=sort_key_id):
        item = compute_item_metrics(
            model=model,
            id_=id_,
            original_row=original_rows[id_],
            perturbed_row=perturbed_rows[id_],
            control_row=control_rows[id_] if control_rows is not None else None,
            metric=metric,
        )
        if item is not None:
            items.append(item)

    return items


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_items(
    model: str,
    items: List[Dict[str, Any]],
    quality_thresholds: Sequence[float],
) -> Dict[str, Any]:
    rsd = finite_values(item["primary_rsd"] for item in items)
    q = finite_values(item["correct_support_rsd"] for item in items)

    correct_items = [item for item in items if item["is_correct"] is True]
    wrong_items = [item for item in items if item["is_correct"] is False]
    known_gold_items = [item for item in items if item["is_correct"] is not None]

    rsd_correct = finite_values(item["primary_rsd"] for item in correct_items)
    rsd_wrong = finite_values(item["primary_rsd"] for item in wrong_items)

    confs = finite_values(item["orig_confidence_abs_margin"] for item in items)

    flips = [item["flipped"] for item in items if item["flipped"] is not None]
    flip_rate = (sum(1 for x in flips if x) / len(flips)) if flips else None

    accuracy = None
    if known_gold_items:
        accuracy = sum(1 for item in known_gold_items if item["is_correct"] is True) / len(known_gold_items)

    sd = stdev(rsd)
    mean_rsd = mean(rsd)
    mean_over_std = None
    if mean_rsd is not None and sd is not None and sd > 0:
        mean_over_std = mean_rsd / sd

    summary = {
        "model": model,
        "n": len(items),
        "n_with_gold": len(known_gold_items),
        "accuracy": accuracy,
        "flip_rate": flip_rate,
        "mean_confidence": mean(confs),
        "faithfulness_mean": mean_rsd,
        "faithfulness_median": median(rsd),
        "faithfulness_std": sd,
        "faithfulness_mean_over_std": mean_over_std,
        "faithfulness_p_positive": p_positive(rsd),
        "faithfulness_correct_mean": mean(rsd_correct),
        "faithfulness_wrong_mean": mean(rsd_wrong),
        "quality_mean": mean(q),
        "quality_median": median(q),
        "quality_p_positive": p_positive(q),  # same as P(q>0)
    }

    # Thresholded Correct-Support Rates: P(q > tau).
    # q already equals max(faithfulness, 0) for correct predictions and 0 for wrong predictions.
    # Therefore P(q > tau) is a frequency-based quality proxy.
    for tau in quality_thresholds:
        summary[threshold_key("quality_p", tau)] = p_greater_than(q, tau)
        summary[threshold_key("quality_n", tau)] = n_greater_than(q, tau)

    return summary


def print_summary(
    summaries: List[Dict[str, Any]],
    metric: str,
    adjusted: bool,
    quality_thresholds: Sequence[float],
) -> None:
    score_name = ("Adjusted " if adjusted else "") + ("MarginRSD" if metric == "margin" else "LogitRSD")
    q_cols = [(tau, threshold_key("quality_p", tau), threshold_key("quality_n", tau)) for tau in quality_thresholds]

    print()
    print("Reasoning Faithfulness / Quality Analysis")
    print(f"Primary faithfulness score: {score_name}")
    print("Correct-Support thresholds: " + ", ".join(f"q>{fmt_threshold(tau)}" for tau in quality_thresholds))

    base_width = 141
    width = base_width + 18 * len(q_cols)
    print("-" * width)
    header = (
        f"{'model':<24} "
        f"{'N':>5} "
        f"{'acc':>8} "
        f"{'flip':>8} "
        f"{'conf':>9} "
        f"{'faith mean':>12} "
        f"{'faith med':>11} "
        f"{'mean/std':>10} "
        f"{'P(f>0)':>9} "
        f"{'faith correct':>14} "
        f"{'faith wrong':>12} "
        f"{'quality mean':>13}"
    )
    for tau, _pkey, _nkey in q_cols:
        header += f" {('P(q>' + fmt_threshold(tau) + ')'):>10} {('N(q>' + fmt_threshold(tau) + ')'):>6}"
    print(header)
    print("-" * width)

    # Sort by quality first if available, otherwise faithfulness.
    def sort_value(r: Dict[str, Any]) -> float:
        if r["quality_mean"] is not None:
            return float(r["quality_mean"])
        if r["faithfulness_mean"] is not None:
            return float(r["faithfulness_mean"])
        return float("-inf")

    for r in sorted(summaries, key=sort_value, reverse=True):
        row = (
            f"{r['model']:<24} "
            f"{r['n']:>5d} "
            f"{fmt(r['accuracy']):>8} "
            f"{fmt(r['flip_rate']):>8} "
            f"{fmt(r['mean_confidence']):>9} "
            f"{fmt(r['faithfulness_mean'], signed=True):>12} "
            f"{fmt(r['faithfulness_median'], signed=True):>11} "
            f"{fmt(r['faithfulness_mean_over_std'], signed=True):>10} "
            f"{fmt(r['faithfulness_p_positive']):>9} "
            f"{fmt(r['faithfulness_correct_mean'], signed=True):>14} "
            f"{fmt(r['faithfulness_wrong_mean'], signed=True):>12} "
            f"{fmt(r['quality_mean'], signed=True):>13}"
        )
        for _tau, pkey, nkey in q_cols:
            row += f" {fmt(r.get(pkey)):>10} {r.get(nkey, 0):>6d}"
        print(row)

    print("-" * width)
    print()
    print("Definitions")
    print("  faithfulness = primary RSD.")
    print("    Larger values mean removing the selected reasoning segment reduces the model's support margin/logit.")
    print("  quality mean = mean CorrectSupportRSD.")
    print("    CorrectSupportRSD = max(faithfulness, 0) if the original prediction is correct; 0 if wrong.")
    print("  P(q>tau) = thresholded Correct-Support Rate.")
    print("    It measures the fraction of examples where the model is correct and has quality score above tau.")
    print("  faith correct / faith wrong are reported separately because faithful wrong reasoning is not high-quality reasoning.")
    print()


# ---------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------

def write_items_csv(path: str, items: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "id",
        "image",
        "reason",
        "output",
        "pred",
        "gold",
        "is_correct",
        "perturbed_pred",
        "flipped",
        "support_label",
        "orig_yes_logit",
        "orig_no_logit",
        "pert_yes_logit",
        "pert_no_logit",
        "orig_confidence_abs_margin",
        "logit_rsd_raw",
        "logit_rsd_adjusted",
        "margin_rsd_raw",
        "margin_rsd_adjusted",
        "primary_rsd",
        "faithfulness",
        "correct_support_rsd",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({k: item.get(k, "") for k in fieldnames})

    print(f"Item CSV saved: {path} ({len(items)} rows)")


def write_summary_csv(
    path: str,
    summaries: List[Dict[str, Any]],
    quality_thresholds: Sequence[float],
) -> None:
    fieldnames = [
        "model",
        "n",
        "n_with_gold",
        "accuracy",
        "flip_rate",
        "mean_confidence",
        "faithfulness_mean",
        "faithfulness_median",
        "faithfulness_std",
        "faithfulness_mean_over_std",
        "faithfulness_p_positive",
        "faithfulness_correct_mean",
        "faithfulness_wrong_mean",
        "quality_mean",
        "quality_median",
        "quality_p_positive",
    ]
    for tau in quality_thresholds:
        fieldnames.append(threshold_key("quality_p", tau))
        fieldnames.append(threshold_key("quality_n", tau))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summaries:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Summary CSV saved: {path} ({len(summaries)} rows)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def compute_global_common_ids(
    common_files: List[str],
    original_dir: str,
    perturbed_dir: str,
    control_dir: Optional[str],
) -> set[str]:
    global_ids: Optional[set[str]] = None

    for fname in common_files:
        orig_rows = load_jsonl(os.path.join(original_dir, fname))
        pert_rows = load_jsonl(os.path.join(perturbed_dir, fname))
        ids = set(orig_rows) & set(pert_rows)

        if control_dir is not None:
            ctrl_rows = load_jsonl(os.path.join(control_dir, fname))
            ids &= set(ctrl_rows)

        if global_ids is None:
            global_ids = ids
        else:
            global_ids &= ids

    return global_ids if global_ids is not None else set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original_dir", help="Directory with original/full-context .jsonl predictions.")
    parser.add_argument("perturbed_dir", help="Directory with perturbed/truncated .jsonl predictions.")
    parser.add_argument("control_dir", nargs="?", default=None, help="Optional control/unrelated .jsonl predictions.")
    parser.add_argument(
        "--metric",
        choices=["margin", "logit"],
        default="margin",
        help="Primary faithfulness metric. Default: margin.",
    )
    parser.add_argument(
        "--common-ids",
        action="store_true",
        help="Use only IDs shared by all methods/files. Useful when you want summaries over identical examples.",
    )
    parser.add_argument("--csv", default=None, help="Optional per-item CSV output path.")
    parser.add_argument("--summary-csv", default=None, help="Optional summary CSV output path.")
    parser.add_argument(
        "--quality-thresholds",
        default="0,0.25,0.5,0.75",
        help="Comma-separated thresholds for P(q>tau). Default: 0,0.25,0.5,0.75.",
    )
    args = parser.parse_args()

    quality_thresholds = parse_thresholds(args.quality_thresholds)

    orig_files = set(jsonl_files(args.original_dir))
    pert_files = set(jsonl_files(args.perturbed_dir))
    common_files = sorted(orig_files & pert_files)

    if args.control_dir is not None:
        ctrl_files = set(jsonl_files(args.control_dir))
        common_files = sorted(set(common_files) & ctrl_files)

    if not common_files:
        raise RuntimeError("No common .jsonl files found across the provided directories.")

    forced_ids = None
    if args.common_ids:
        forced_ids = compute_global_common_ids(
            common_files=common_files,
            original_dir=args.original_dir,
            perturbed_dir=args.perturbed_dir,
            control_dir=args.control_dir,
        )
        print(f"Using global common ID set: {len(forced_ids)} examples")

    all_items: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for fname in common_files:
        model = fname.replace(".jsonl", "")

        orig_rows = load_jsonl(os.path.join(args.original_dir, fname))
        pert_rows = load_jsonl(os.path.join(args.perturbed_dir, fname))
        ctrl_rows = load_jsonl(os.path.join(args.control_dir, fname)) if args.control_dir is not None else None

        items = collect_items_for_model(
            model=model,
            original_rows=orig_rows,
            perturbed_rows=pert_rows,
            control_rows=ctrl_rows,
            metric=args.metric,
            forced_ids=forced_ids,
        )

        all_items.extend(items)
        summaries.append(summarize_items(model, items, quality_thresholds=quality_thresholds))

    print_summary(
        summaries=summaries,
        metric=args.metric,
        adjusted=args.control_dir is not None,
        quality_thresholds=quality_thresholds,
    )

    if args.csv:
        write_items_csv(args.csv, all_items)

    if args.summary_csv:
        write_summary_csv(args.summary_csv, summaries, quality_thresholds=quality_thresholds)


if __name__ == "__main__":
    main()
