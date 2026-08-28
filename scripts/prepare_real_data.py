"""
ReturnGuard Agent - real-data adapter for the Olist Brazilian E-Commerce
dataset (Kaggle: olistbr/brazilian-ecommerce).

This is the SECOND data path. It maps the seven raw Olist CSVs into the
exact same unified schema scripts/generate_data.py produces for the
synthetic path, so the rest of the pipeline (src/features.py,
src/model.py, scripts/train.py, evaluation/*) is dataset-agnostic.

It NEVER touches data/synthetic/ or models/xgboost_model_synthetic.joblib.

The mapping decisions below are already validated - do not re-derive them:

  * Keep only orders that have a review (~99.8% coverage across all order
    statuses; inner join orders -> reviews on order_id). Reviews are
    de-duplicated to one per order first (a few order_ids have 2 review
    rows) by keeping the earliest review_creation_date.

  * LABEL  was_returned = review_score <= 2. A documented proxy: Olist
    has no real return flag, only post-purchase review scores.

  * order_amount = sum of payment_value over all payment_sequential rows
    for the order (olist_order_payments_dataset).

  * item_count = number of rows for the order in
    olist_order_items_dataset.

  * category = English-translated product category of the HIGHEST-PRICE
    item in the order. Missing / untranslated -> "unknown".

  * size_flag = category in the seven Olist fashion categories (note the
    real typo "fashio_female_clothing" in Olist's own translation table
    is kept as-is).

  * payment_method = payment_type of the payment_sequential == 1 row,
    recoded: boleto -> "COD" (closest real analog: cash-based,
    pre-shipment bank slip - NOT true cash-on-delivery), credit_card /
    debit_card -> "card", voucher -> "voucher". Orders whose
    sequential-1 payment_type is "not_defined" are dropped.

  * customer_id = customer_unique_id (the real person; Olist's per-order
    customer_id is NOT stable across a person's orders).

  * is_new_customer / days_since_signup / customer_prior_return_rate are
    computed PER CUSTOMER, chronologically, using only that customer's
    own strictly-earlier orders. customer_prior_return_rate is an
    expanding mean of was_returned over prior orders (shift(1) so the
    current order's own label is excluded).

  * discount_pct DOES NOT EXIST in Olist. Approach (b): the column is
    kept in the unified schema but filled NaN for every real row, so it
    is imputed away and contributes no signal. Synthetic keeps its real
    values untouched.

  The following three are computed AFTER the temporal split, because they
  are fit on the train split only:

  * delivery_pincode_risk_tier: each customer_state bucketed into
    low/medium/high tertiles of that state's historical late-delivery
    rate (late = order_delivered_customer_date is null OR later than
    order_estimated_delivery_date). Tertile cutpoints fit on TRAIN rows
    only; val/test use the same lookup; unseen states -> "medium".

  * review_score_avg: redefined as the PRODUCT's average review score
    (highest-price item's product_id), because only ~3% of Olist
    customers are repeat buyers. Train rows use a leave-one-out mean
    (their own review excluded); val/test rows use the plain train-period
    product mean. Products with no other train reviews -> NaN (imputer
    handles it).

Usage:
    python -m scripts.prepare_real_data --raw-dir data/olist_raw --out-dir data/real
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# Reuse the synthetic path's chronological split - do NOT duplicate it.
from scripts.generate_data import temporal_split

# Olist's own (typo-preserving) fashion category slugs.
FASHION_CATEGORIES = {
    "fashio_female_clothing",
    "fashion_bags_accessories",
    "fashion_childrens_clothes",
    "fashion_male_clothing",
    "fashion_shoes",
    "fashion_sport",
    "fashion_underwear_beach",
}

PAYMENT_TYPE_RECODE = {
    "boleto": "COD",
    "credit_card": "card",
    "debit_card": "card",
    "voucher": "voucher",
}

FINAL_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "order_amount",
    "category",
    "item_count",
    "discount_pct",  # approach (b): present, all-NaN for real
    "payment_method",
    "is_new_customer",
    "customer_prior_return_rate",
    "days_since_signup",
    "delivery_pincode_risk_tier",
    "size_flag",
    "review_score_avg",
    "was_returned",
]


def _read(raw_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(raw_dir, name))


def build_order_level(raw_dir: str) -> pd.DataFrame:
    """Everything that does NOT depend on the temporal split."""
    orders = _read(raw_dir, "olist_orders_dataset.csv")
    reviews = _read(raw_dir, "olist_order_reviews_dataset.csv")
    payments = _read(raw_dir, "olist_order_payments_dataset.csv")
    items = _read(raw_dir, "olist_order_items_dataset.csv")
    products = _read(raw_dir, "olist_products_dataset.csv")
    customers = _read(raw_dir, "olist_customers_dataset.csv")
    translation = _read(raw_dir, "product_category_name_translation.csv")

    # --- one review per order (earliest), then inner-join to orders ---
    reviews = reviews.sort_values("review_creation_date")
    reviews = reviews.drop_duplicates(subset="order_id", keep="first")
    df = orders.merge(reviews[["order_id", "review_score"]], on="order_id", how="inner")

    df["was_returned"] = df["review_score"] <= 2

    # --- order_amount: sum of every payment row for the order ---
    order_amount = payments.groupby("order_id")["payment_value"].sum().rename("order_amount")
    df = df.merge(order_amount, on="order_id", how="left")

    # --- payment_method: the payment_sequential == 1 row, recoded ---
    seq1 = payments[payments["payment_sequential"] == 1][["order_id", "payment_type"]]
    seq1 = seq1.drop_duplicates(subset="order_id", keep="first")
    df = df.merge(seq1, on="order_id", how="left")
    df = df[df["payment_type"] != "not_defined"]
    df["payment_method"] = df["payment_type"].map(PAYMENT_TYPE_RECODE)
    # any payment_type not in the recode map (or a missing seq-1 row) -> drop,
    # since payment_method is a required categorical feature.
    df = df[df["payment_method"].notna()]

    # --- item_count + highest-price item's product / category ---
    item_count = items.groupby("order_id").size().rename("item_count")
    df = df.merge(item_count, on="order_id", how="left")

    top_item = (
        items.sort_values("price", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")[["order_id", "product_id"]]
    )
    df = df.merge(top_item, on="order_id", how="left")

    cat = products[["product_id", "product_category_name"]].merge(
        translation, on="product_category_name", how="left"
    )
    df = df.merge(
        cat[["product_id", "product_category_name_english"]], on="product_id", how="left"
    )
    df["category"] = df["product_category_name_english"].fillna("unknown")
    df["size_flag"] = df["category"].isin(FASHION_CATEGORIES)

    # --- customer_id = customer_unique_id (the real person) ---
    df = df.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left",
    )
    df = df.rename(columns={"customer_id": "olist_order_customer_id", "customer_unique_id": "customer_id"})

    # --- order_date / timestamp ---
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_date"] = df["order_purchase_timestamp"].dt.date.astype(str)

    # --- late-delivery flag (used later, after the split, for the tier) ---
    delivered = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    estimated = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    df["_is_late"] = delivered.isna() | (delivered > estimated)

    # drop rows with no order_amount or no items (can't score them)
    df = df[df["order_amount"].notna() & df["item_count"].notna()]

    # discount_pct: does not exist in Olist -> all-NaN (approach (b))
    df["discount_pct"] = np.nan

    return df.reset_index(drop=True)


def add_per_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    """is_new_customer, days_since_signup, customer_prior_return_rate -
    each computed from ONLY that customer's own strictly-earlier orders."""
    df = df.sort_values(["customer_id", "order_purchase_timestamp"]).copy()
    g = df.groupby("customer_id", sort=False)

    order_rank = g.cumcount()
    df["is_new_customer"] = order_rank == 0

    first_purchase = g["order_purchase_timestamp"].transform("min")
    df["days_since_signup"] = (df["order_purchase_timestamp"] - first_purchase).dt.days

    # expanding mean of was_returned over PRIOR orders only (shift by 1)
    prior_rate = (
        g["was_returned"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["customer_prior_return_rate"] = prior_rate

    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


def fit_state_risk_tiers(train_df: pd.DataFrame) -> dict:
    """low/medium/high by tertiles of each state's train-period
    late-delivery rate."""
    rate = train_df.groupby("customer_state")["_is_late"].mean()
    if rate.empty:
        return {}
    q1, q2 = rate.quantile([1 / 3, 2 / 3])

    def tier(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "medium"
        return "high"

    return {state: tier(v) for state, v in rate.items()}


def apply_state_risk_tiers(df: pd.DataFrame, lookup: dict) -> pd.Series:
    return df["customer_state"].map(lookup).fillna("medium")


def add_product_review_avg(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """review_score_avg = product's average review score.

    train: leave-one-out (exclude the row's own review).
    val/test: plain train-period product average.
    products with no other train reviews -> NaN.
    """
    grp = train.groupby("product_id")["review_score"]
    p_sum = grp.transform("sum")
    p_cnt = grp.transform("count")
    loo = (p_sum - train["review_score"]) / (p_cnt - 1)
    train = train.copy()
    train["review_score_avg"] = loo  # NaN where p_cnt == 1

    train_avg = train.groupby("product_id")["review_score"].mean()
    for frame in (val, test):
        frame["review_score_avg"] = frame["product_id"].map(train_avg)

    return train, val.copy(), test.copy()


def prepare(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order_level = build_order_level(raw_dir)
    order_level = add_per_customer_history(order_level)

    train, val, test = temporal_split(order_level)

    # --- post-split, train-fit-only features ---
    tier_lookup = fit_state_risk_tiers(train)
    for frame in (train, val, test):
        frame["delivery_pincode_risk_tier"] = apply_state_risk_tiers(frame, tier_lookup)

    train, val, test = add_product_review_avg(train, val, test)

    out = []
    for frame in (train, val, test):
        out.append(frame[FINAL_COLUMNS].reset_index(drop=True))
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Olist real-data path")
    parser.add_argument("--raw-dir", type=str, default="data/olist_raw")
    parser.add_argument("--out-dir", type=str, default="data/real")
    args = parser.parse_args()

    train, val, test = prepare(args.raw_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    total = len(train) + len(val) + len(test)
    print(f"Olist real data -> {total} orders: {len(train)} train / {len(val)} val / {len(test)} test")
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        print(
            f"  {name}: return rate {frame['was_returned'].mean():.3f}  |  "
            f"new customers {frame['is_new_customer'].mean():.3f}  |  "
            f"missing review_score_avg {frame['review_score_avg'].isna().mean():.3f}  |  "
            f"missing prior_return_rate {frame['customer_prior_return_rate'].isna().mean():.3f}"
        )
    print(f"Wrote train.csv, val.csv, test.csv to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
