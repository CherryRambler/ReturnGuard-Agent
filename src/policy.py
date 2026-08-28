"""
ReturnGuard Agent - decision policy.

Pure(ish), deterministic mapping from a model's risk score and order
context to ONE of three bounded actions. This is the module that turns
a risk *scorer* into a risk *agent*: every action is capped, reversible,
and meant to be logged by the caller (see backend/db.py's audit_log table).

See: razorpay_buildathon_plan.md, Section 6 (Technical Architecture,
"Decision & Action Layer") and Section 7 ("audit_log" table).

Design principles:
  - No network calls, no ML inside this module - it only ever sees a
    risk_score float and a few order fields. That's what makes it
    trivially unit-testable in isolation from the model (see
    tests/test_policy.py) and easy to defend in a pitch: "the model
    can be wrong, but the policy can't misbehave in ways we haven't
    reviewed."
  - Exactly three possible actions. Adding a fourth is a deliberate,
    reviewed code change - not something a threshold tweak can invent.
  - restrict_cod only ever applies to COD orders. Applying it to a
    card/UPI/netbanking order is meaningless (there's no COD to
    restrict) - a high-risk non-COD order is flagged for review instead.
  - The hourly cap on restrict_cod exists so a bad batch of high scores
    (a bug, a data drift, a coordinated attack) can't restrict an
    unbounded number of real customers before a human notices. When the
    cap is hit, the action is downgraded to flag_for_review - never
    silently dropped, never silently escalated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

ALLOW = "allow"
RESTRICT_COD = "restrict_cod"
FLAG_FOR_REVIEW = "flag_for_review"

VALID_ACTIONS = {ALLOW, RESTRICT_COD, FLAG_FOR_REVIEW}


@dataclass
class Decision:
    """The output of a policy call - everything the caller needs to act on
    and to write a complete audit_log row."""

    action: str
    risk_score: float
    capped: bool = False  # True if this action was downgraded by the hourly cap
    reason: str = ""      # short human-readable summary (paired with SHAP reasons by the caller)


@dataclass
class DecisionPolicy:
    """Stateful wrapper around the underlying decision rule.

    Tracks a per-hour count of restrict_cod actions so the agent can
    never restrict more than `max_restrict_cod_per_hour` orders within
    a rolling hour bucket. Everything else about the policy is a pure
    function of its inputs - the state exists only for the cap.
    """

    allow_threshold: float = 0.35        # risk_score below this -> allow
    restrict_threshold: float = 0.65     # risk_score at/above this (and COD) -> restrict_cod
    max_restrict_cod_per_hour: int = 20

    _hourly_counts: dict = field(default_factory=lambda: defaultdict(int), repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.allow_threshold < self.restrict_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= allow_threshold < restrict_threshold <= 1, got "
                f"allow_threshold={self.allow_threshold}, restrict_threshold={self.restrict_threshold}"
            )

    def reset(self) -> None:
        """Clear the hourly counters. Mainly useful for tests."""
        self._hourly_counts.clear()

    def decide(
        self,
        risk_score: float,
        payment_method: str,
        timestamp: Optional[datetime] = None,
    ) -> Decision:
        if not 0.0 <= risk_score <= 1.0:
            raise ValueError(f"risk_score must be in [0, 1], got {risk_score}")

        timestamp = timestamp or datetime.now(timezone.utc)
        base_action, reason = self._base_decision(risk_score, payment_method)

        if base_action != RESTRICT_COD:
            return Decision(action=base_action, risk_score=risk_score, capped=False, reason=reason)

        # restrict_cod is the only action subject to the hourly cap.
        bucket = self._hour_bucket(timestamp)
        if self._hourly_counts[bucket] >= self.max_restrict_cod_per_hour:
            return Decision(
                action=FLAG_FOR_REVIEW,
                risk_score=risk_score,
                capped=True,
                reason=(
                    f"{reason} (downgraded: hourly restrict_cod cap of "
                    f"{self.max_restrict_cod_per_hour} already reached)"
                ),
            )

        self._hourly_counts[bucket] += 1
        return Decision(action=RESTRICT_COD, risk_score=risk_score, capped=False, reason=reason)

    def _base_decision(self, risk_score: float, payment_method: str) -> tuple[str, str]:
        """The uncapped decision, before the hourly-cap check is applied."""
        if risk_score < self.allow_threshold:
            return ALLOW, f"risk score {risk_score:.2f} below allow threshold ({self.allow_threshold})"

        if risk_score >= self.restrict_threshold:
            if payment_method == "COD":
                return (
                    RESTRICT_COD,
                    f"risk score {risk_score:.2f} >= restrict threshold ({self.restrict_threshold}); COD order",
                )
            return (
                FLAG_FOR_REVIEW,
                f"risk score {risk_score:.2f} >= restrict threshold ({self.restrict_threshold}); "
                "non-COD order, so flagged instead of restricted",
            )

        return (
            FLAG_FOR_REVIEW,
            f"risk score {risk_score:.2f} between thresholds ({self.allow_threshold}-{self.restrict_threshold})",
        )

    @staticmethod
    def _hour_bucket(timestamp: datetime) -> str:
        return timestamp.strftime("%Y-%m-%dT%H")


def decide_action(
    risk_score: float,
    payment_method: str,
    policy: Optional[DecisionPolicy] = None,
) -> Decision:
    """Convenience function for one-off calls without managing a
    DecisionPolicy instance yourself. Note this creates a fresh policy
    (and therefore a fresh hourly-cap counter) on every call unless you
    pass one in - the inference API should hold one shared DecisionPolicy
    instance for the cap to mean anything across requests."""
    policy = policy or DecisionPolicy()
    return policy.decide(risk_score, payment_method)
