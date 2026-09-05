"""
ReturnGuard Agent - agent trust evaluation (human-override rate).

TODO (Day 11, per docs/buildathon_plan.md Section 8):
  1. Run the full held-out test set through src/policy.py's DecisionPolicy
     so every order gets an "auto_action" row (write these to audit_log
     via backend/db.py, or a standalone sqlite file for this script).
  2. Sample ~50 of those actions, weighted toward restrict_cod and
     flag_for_review (allow is low-stakes, sample fewer of those).
  3. Manually review each sampled action (playing the role of a
     reviewing ops analyst) and record keep/override in a small CSV or
     inline dict - this part is genuinely manual, not automatable.
  4. override_rate = (# you'd override) / (# reviewed), reported
     separately per action type. Append this to evaluation/eval_report.md
     next to the classifier metrics, not as a replacement for them.

This script's output is a human-in-the-loop artifact - expect to run it
interactively rather than fully unattended.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the agent's human-override trust rate")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--out", type=str, default=None,
                        help="defaults to evaluation/eval_report_<dataset>.md")
    args = parser.parse_args()

    raise NotImplementedError("Implement per the module docstring on Day 11 - this one is manual-review-driven.")


if __name__ == "__main__":
    main()
