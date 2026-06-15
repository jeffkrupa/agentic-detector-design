"""Score a model's sensitivity predictions against AD ground truth.

Implements the protocol in docs/BENCHMARK.md: the model predicts the *sign* and
*log10 magnitude* (+ confidence) of dO/dtheta from the physical setup alone
(closed-book) or with a gradient-tool query budget (grounded). We report sign
accuracy, magnitude MAE in dex, within-1-dex rate, calibration (ECE), all
stratified by the ground-truth reliability flag.

    python benchmark/run_benchmark.py --dataset benchmark/data/quick.jsonl --model dry-run

The ``dry-run`` model is a transparent physics-prior baseline so the scorer runs
offline; plug a real model into ``predict_with_model`` (TODO marked).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Coarse textbook priors for an EM sampling calorimeter (electron primary).
# These are intentionally simple; they are the closed-book baseline, not truth.
_PRIOR_SIGN = {
    # observable, parameter -> expected sign
    ("total_edep", "energy"): "+",       # more beam energy -> more deposit
    ("total_edep", "a"): "+",            # thicker absorber -> more sampled deposit (regime-dependent)
    ("total_edep", "g"): "+",
    ("visible_fraction", "energy"): "-", # deposit grows slower than linearly / leakage
    ("shower_max_depth", "energy"): "+", # shower max ~ ln(E): deeper with energy
    ("shower_max_depth", "a"): "-",      # thicker absorber -> earlier (shallower layer index)
    ("peak_edep", "energy"): "+",
    ("front_fraction", "a"): "+",        # thicker absorber -> more up-front deposition
}


def predict_dry_run(item: dict) -> dict:
    """Transparent physics-prior baseline: prior sign, magnitude ~ value scale."""
    o, wrt = item["observable"], item["wrt"]
    sign = _PRIOR_SIGN.get((o, wrt), "~0")
    # crude magnitude guess: assume dO/dtheta ~ O / theta scale (dimensional prior)
    dp = item["design_point"]
    theta = {"a": dp["a"], "g": dp["g"], "energy": dp["energy"]}[wrt]
    # we are not allowed to see ground truth; guess log10 from a unit-ish prior
    guess_log10 = math.log10(max(abs(theta), 1.0)) - 1.0
    return {"sign": sign, "log10_abs": guess_log10, "confidence": 0.5}


def predict_with_model(item: dict, model: str) -> dict:
    """TODO(claude-code): query a real LLM with the physical setup only (no truth).

    Build a prompt from item['observable'], item['wrt'], item['design_point'],
    and the geometry description; require a JSON answer
    {"sign": "+|-|~0", "log10_abs": <float>, "confidence": <0..1>, "why": <str>}.
    For the *grounded* condition, expose the `sensitivity` tool with a query
    budget and let the model call it on a subset of items.
    """
    raise NotImplementedError("Wire a real model; see CLAUDE.md task #8.")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _sign_correct(pred_sign: str, gt_sign: str) -> bool:
    return pred_sign == gt_sign


def score(items, preds) -> dict:
    buckets = {"all": [], "ok": [], "marginal": [], "untrusted": []}
    by_obs = {}
    for it, pr in zip(items, preds):
        gt = it["ground_truth"]
        rec = {
            "sign_ok": _sign_correct(pr["sign"], gt["sign"]),
            "dex_err": (abs(pr["log10_abs"] - gt["log10_abs"])
                        if (pr.get("log10_abs") is not None and gt.get("log10_abs") is not None)
                        else None),
            "confidence": pr.get("confidence", 0.5),
            "reliability": gt["reliability"],
        }
        buckets["all"].append(rec)
        buckets[gt["reliability"]].append(rec)
        by_obs.setdefault(it["observable"], []).append(rec)

    def agg(recs):
        if not recs:
            return {"n": 0}
        sign_acc = sum(r["sign_ok"] for r in recs) / len(recs)
        dex = [r["dex_err"] for r in recs if r["dex_err"] is not None]
        mae = sum(dex) / len(dex) if dex else None
        within1 = (sum(d <= 1.0 for d in dex) / len(dex)) if dex else None
        ece = _ece(recs)
        return {"n": len(recs), "sign_accuracy": round(sign_acc, 4),
                "mag_mae_dex": None if mae is None else round(mae, 4),
                "within_1_dex": None if within1 is None else round(within1, 4),
                "ece": round(ece, 4)}

    return {
        "overall": agg(buckets["all"]),
        "by_reliability": {k: agg(buckets[k]) for k in ("ok", "marginal", "untrusted")},
        "by_observable": {k: agg(v) for k, v in by_obs.items()},
    }


def _ece(recs, n_bins: int = 10) -> float:
    """Expected calibration error of confidence vs sign-correctness."""
    bins = [[] for _ in range(n_bins)]
    for r in recs:
        b = min(n_bins - 1, int(r["confidence"] * n_bins))
        bins[b].append(r)
    ece, total = 0.0, len(recs)
    for b, group in enumerate(bins):
        if not group:
            continue
        conf = sum(r["confidence"] for r in group) / len(group)
        acc = sum(r["sign_ok"] for r in group) / len(group)
        ece += (len(group) / total) * abs(conf - acc)
    return ece


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="dry-run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    items = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()]
    if args.model == "dry-run":
        preds = [predict_dry_run(it) for it in items]
    else:
        preds = [predict_with_model(it, args.model) for it in items]

    results = score(items, preds)
    results["model"] = args.model
    results["dataset"] = args.dataset
    results["n_items"] = len(items)

    out = Path(args.out) if args.out else Path("benchmark/results") / (
        Path(args.dataset).stem + f"__{args.model.replace(':', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["overall"], indent=2))
    print(f"[run_benchmark] full results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
