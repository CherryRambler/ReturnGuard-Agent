"""
ReturnGuard Agent - cost-based threshold selection.

Turns a calibrated risk_score into the two thresholds src/policy.py's
DecisionPolicy needs (allow_threshold, restrict_threshold), using the
false-positive/false-negative cost figures from Section 8 of the plan -
not just whatever threshold happens to maximize F1.

allow_threshold: the point below which acting at all isn't worth it -
    chosen to maximize net value (value of returns caught minus the
    friction cost of false positives) on the validation set.

restrict_threshold: a stricter, precision-gated cutoff for restrict_cod
    specifically, since it's the one customer-facing action. Only apply
    it where we're confident enough (>= min_precision) among COD orders
    - "some net benefit on average" isn't a high enough bar for an
    action a real customer feels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

COST_PER_FALSE_POSITIVE = 50   # INR - friction cost of blocking/reviewing a good order
COST_PER_FALSE_NEGATIVE = 250  # INR - avg loss from a missed return (shipping + restocking + margin)


@dataclass
class ThresholdRow:
    threshold: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    net_value: float
    flag_rate: float  # fraction of ALL orders that get some action (not just "allow")


def sweep_thresholds(
    y_true,
    scores,
    thresholds: Optional[np.ndarray] = None,
    cost_fp: float = COST_PER_FALSE_POSITIVE,
    cost_fn: float = COST_PER_FALSE_NEGATIVE,
) -> list[ThresholdRow]:
    """One ThresholdRow per candidate threshold, so the full cost/precision
    tradeoff curve is inspectable - not just whichever point wins."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    total = len(y_true)
    rows = []
    for t in thresholds:
        preds = scores >= t
        tp = int(np.sum(preds & y_true))
        fp = int(np.sum(preds & ~y_true))
        fn = int(np.sum(~preds & y_true))
        tn = int(np.sum(~preds & ~y_true))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        net_value = tp * cost_fn - fp * cost_fp
        flag_rate = (tp + fp) / total if total else 0.0
        rows.append(ThresholdRow(float(t), precision, recall, tp, fp, fn, tn, net_value, flag_rate))
    return rows


def select_allow_threshold(
    y_true,
    scores,
    min_precision: float = 0.40,
    max_flag_rate: float = 0.40,
    **cost_kwargs,
) -> ThresholdRow:
    """The net-value-maximizing threshold, WITH two guardrails:

    1. min_precision: pure net-value maximization can degenerate into
       flagging almost every order when cost_fn >> cost_fp - "worth
       flagging" needs to stay meaningfully better than chance.
    2. max_flag_rate: even a precision-respectable threshold can still
       flag an operationally unworkable fraction of all orders (e.g.
       60%+) if returns are common in the data. A real review team has
       finite capacity - max_flag_rate caps how much of the order volume
       the policy is allowed to route to a human, forcing the threshold
       to focus on the highest-value cases rather than "most things that
       look even slightly risky."

    Both constraints are applied together first; if nothing clears both,
    the flag-rate cap relaxes first (precision matters more for trust
    than raw queue size), then the precision floor as a last resort -
    each relaxation is printed by the caller so it's never silent."""
    rows = sweep_thresholds(y_true, scores, **cost_kwargs)

    both = [r for r in rows if r.precision >= min_precision and r.flag_rate <= max_flag_rate]
    if both:
        return max(both, key=lambda r: r.net_value)

    precision_only = [r for r in rows if r.precision >= min_precision]
    if precision_only:
        return max(precision_only, key=lambda r: r.net_value)

    return max(rows, key=lambda r: r.net_value)


def select_restrict_threshold(
    y_true_cod,
    scores_cod,
    min_threshold: float,
    min_precision: float = 0.65,
) -> float:
    """Smallest threshold >= min_threshold where precision among COD
    orders reaches min_precision. Falls back to min_threshold + 0.15
    (capped at 0.95) if nothing in the grid reaches that bar - better to
    be explicit about the fallback than silently pick something that
    doesn't actually satisfy the target."""
    rows = sweep_thresholds(y_true_cod, scores_cod, thresholds=np.arange(min_threshold, 0.96, 0.01))
    qualifying = [r for r in rows if r.precision >= min_precision]
    if qualifying:
        return min(qualifying, key=lambda r: r.threshold).threshold
    return min(min_threshold + 0.15, 0.95)
