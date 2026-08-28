"""
ReturnGuard Agent - synthetic order data generator.

Generates a synthetic e-commerce order dataset with a NOISY latent
return-risk function, so the two classes are not trivially separable
(a common failure mode in synthetic fraud/return datasets that makes
precision/recall look fake). Produces a temporal train/val/test split
(70/15/15) so the held-out test set behaves like "future" orders and
there is no leakage from random shuffling across time.

See: razorpay_buildathon_plan.md, Section 7 (Dataset Design) for the
full column spec and design rationale.

Usage:
    python scripts/generate_data.py --n-rows 5000 --seed 42 --out-dir data/synthetic
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = ["apparel", "electronics", "home", "beauty", "footwear"]
CATEGORY_WEIGHTS = [0.32, 0.18, 0.20, 0.15, 0.15]

PAYMENT_METHODS = ["UPI", "card", "netbanking", "COD"]
PAYMENT_WEIGHTS = [0.42, 0.28, 0.10, 0.20]

RISK_TIERS = ["low", "medium", "high"]
RISK_TIER_WEIGHTS = [0.60, 0.30, 0.10]
RISK_TIER_FACTOR = {"low": 0.0, "medium": 0.15, "high": 0.35}

SIZE_RELATED_CATEGORIES = {"apparel", "footwear"}

# Weights in the latent risk function. Kept as named constants (not magic
# numbers inline) so Day 11 error analysis can cite exactly which signal
# was over/under-weighted when explaining a failure mode.
W_DISCOUNT = 0.30
W_SIZE_FLAG = 0.25
W_LOW_REVIEW = 0.20
W_PRIOR_RETURN_RATE = 0.15
W_NEW_CUSTOMER = 0.10
W_RISK_TIER = 0.50  # multiplies RISK_TIER_FACTOR, which is already scaled small
NOISE_STD = 0.15  # this is the knob that controls how separable the classes are


def _make_customer_pool(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    """Each customer gets a persistent 'latent return propensity' in [0, 1].

    This is what makes customer_prior_return_rate real signal rather than
    noise: repeat customers behave consistently across their own orders,
    which a model can actually learn from.
    """
    propensity = np.clip(rng.beta(2, 6, size=n_customers), 0, 1)  # right-skewed: most customers low-risk
    return pd.DataFrame(
        {
            "customer_id": [f"cust_{i:06d}" for i in range(n_customers)],
            "latent_propensity": propensity,
        }
    )


def generate(n_rows: int, seed: int, start_date: str, n_days: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- customer pool: some repeat (Zipf-weighted), some one-time ---
    n_customers = max(int(n_rows * 0.55), 100)
    customers = _make_customer_pool(n_customers, rng)
    zipf_weights = 1.0 / np.arange(1, n_customers + 1)
    zipf_weights /= zipf_weights.sum()
    customer_idx = rng.choice(n_customers, size=n_rows, p=zipf_weights)
    order_customers = customers.iloc[customer_idx].reset_index(drop=True)

    # --- order-level features ---
    start = datetime.fromisoformat(start_date)
    order_offsets = np.sort(rng.integers(0, n_days, size=n_rows))
    order_date = [start + timedelta(days=int(d)) for d in order_offsets]

    category = rng.choice(CATEGORIES, size=n_rows, p=CATEGORY_WEIGHTS)
    order_amount = np.round(np.clip(rng.lognormal(mean=6.8, sigma=0.7, size=n_rows), 150, 15000), 2)
    item_count = rng.integers(1, 7, size=n_rows)  # 1-6 inclusive
    discount_pct = np.round(np.clip(rng.beta(2, 4, size=n_rows) * 60, 0, 60), 1)
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows, p=PAYMENT_WEIGHTS)
    risk_tier = rng.choice(RISK_TIERS, size=n_rows, p=RISK_TIER_WEIGHTS)

    # is_new_customer: True on a customer's chronologically first order in
    # this dataset, False on every later order from the same customer.
    order_seq = pd.DataFrame(
        {"customer_id": order_customers["customer_id"], "order_date": order_date}
    ).reset_index()
    order_seq = order_seq.sort_values("order_date")
    order_seq["is_new"] = ~order_seq.duplicated(subset="customer_id", keep="first")
    is_new_customer = order_seq.sort_values("index")["is_new"].to_numpy()

    days_since_signup = np.where(
        is_new_customer,
        rng.integers(0, 30, size=n_rows),
        rng.integers(30, 700, size=n_rows),
    )

    size_flag = np.isin(category, list(SIZE_RELATED_CATEGORIES))

    # customer_prior_return_rate: null for new customers (no history yet),
    # else their latent propensity plus a little per-order noise.
    prior_return_rate = order_customers["latent_propensity"].to_numpy() + rng.normal(0, 0.05, size=n_rows)
    prior_return_rate = np.clip(prior_return_rate, 0, 1)
    prior_return_rate = np.where(is_new_customer, np.nan, prior_return_rate)

    review_score_avg = np.clip(rng.normal(4.0, 0.6, size=n_rows), 1, 5)
    missing_review_mask = rng.random(n_rows) < 0.05  # ~5% missing, an explicit edge case
    review_score_avg = np.where(missing_review_mask, np.nan, np.round(review_score_avg, 2))

    # --- latent risk function: weighted signal + noise, NOT a hard rule ---
    # Filled-for-risk-calc versions only (the actual columns keep their NaNs
    # so the feature pipeline has to handle missingness for real later).
    review_filled = np.where(np.isnan(review_score_avg), 4.0, review_score_avg)
    prior_filled = np.where(np.isnan(prior_return_rate), 0.25, prior_return_rate)  # category-average-ish fallback
    risk_tier_term = np.array([RISK_TIER_FACTOR[t] for t in risk_tier])

    latent_risk = (
        W_DISCOUNT * (discount_pct / 60.0)
        + W_SIZE_FLAG * size_flag.astype(float)
        + W_LOW_REVIEW * (1 - review_filled / 5.0)
        + W_PRIOR_RETURN_RATE * prior_filled
        + W_NEW_CUSTOMER * is_new_customer.astype(float)
        + W_RISK_TIER * risk_tier_term
    )
    noise = rng.normal(0, NOISE_STD, size=n_rows)
    risk_prob = np.clip(latent_risk + noise, 0.02, 0.95)
    was_returned = rng.random(n_rows) < risk_prob

    df = pd.DataFrame(
        {
            "order_id": [str(uuid.uuid4()) for _ in range(n_rows)],
            "order_date": [d.date().isoformat() for d in order_date],
            "customer_id": order_customers["customer_id"].to_numpy(),
            "order_amount": order_amount,
            "category": category,
            "item_count": item_count,
            "discount_pct": discount_pct,
            "payment_method": payment_method,
            "is_new_customer": is_new_customer,
            "customer_prior_return_rate": prior_return_rate,
            "days_since_signup": days_since_signup,
            "delivery_pincode_risk_tier": risk_tier,
            "size_flag": size_flag,
            "review_score_avg": review_score_avg,
            "was_returned": was_returned,
        }
    )

    # --- inject irreducible-noise edge cases: ~3% of rows get an exact
    #     feature-duplicate whose label is resampled independently, so
    #     some pairs end up with identical features but opposite labels ---
    n_dupes = max(int(0.03 * n_rows), 1)
    dupe_source_idx = rng.choice(n_rows, size=n_dupes, replace=False)
    dupes = df.iloc[dupe_source_idx].copy()
    dupes["order_id"] = [str(uuid.uuid4()) for _ in range(n_dupes)]
    dupe_risk = risk_prob[dupe_source_idx]
    dupes["was_returned"] = rng.random(n_dupes) < dupe_risk
    df = pd.concat([df, dupes], ignore_index=True)

    df = df.sort_values("order_date").reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Split by chronological order (not random) so test = the 'future'.

    This is what Section 7 of the plan means by "leakage prevention":
    the model never sees a test-set order's features during training,
    tuning, or threshold selection - not even indirectly via a random
    shuffle that mixes past and future rows together.
    """
    df = df.sort_values("order_date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ReturnGuard Agent synthetic order data")
    parser.add_argument("--n-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default="2026-01-01")
    parser.add_argument("--n-days", type=int, default=240)
    parser.add_argument("--out-dir", type=str, default="data/synthetic")
    args = parser.parse_args()

    df = generate(args.n_rows, args.seed, args.start_date, args.n_days)
    train, val, test = temporal_split(df)

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    print(f"Generated {len(df)} orders -> {len(train)} train / {len(val)} val / {len(test)} test")
    print(f"Overall return rate: {df['was_returned'].mean():.3f}")
    print(f"  train: {train['was_returned'].mean():.3f}  |  val: {val['was_returned'].mean():.3f}  |  test: {test['was_returned'].mean():.3f}")
    print(f"Missing review_score_avg: {df['review_score_avg'].isna().mean():.3f}")
    print(f"Missing customer_prior_return_rate (new customers): {df['customer_prior_return_rate'].isna().mean():.3f}")
    print(f"Wrote train.csv, val.csv, test.csv to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
