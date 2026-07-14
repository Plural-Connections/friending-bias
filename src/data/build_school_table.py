"""Stage 0 — Data Assembly & Target Construction.

Builds data/processed/schools_master.parquet and
data/interim/reviews_clean.parquet from raw Atlas + GreatSchools inputs.

Run:
    python -m src.data.build_school_table --config configs/stage0.yaml
"""
import argparse

import pandas as pd
import yaml


def crosswalk_to_nces(df: pd.DataFrame, name_col: str, address_col: str, state_col: str) -> pd.DataFrame:
    """Fuzzy-match fallback only if NCES ID absent; log unmatched rate."""
    raise NotImplementedError


def clean_reviews(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Dedup (flag near-duplicates by reviewer/text similarity), strip HTML,
    drop reviews < N words, flag language != en, compute word_count and
    review_year. These columns are required for confounder controls in
    later stages regardless of aggregation strategy chosen.
    """
    raise NotImplementedError


def aggregate_reviews_to_school(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate review-level data to school grain per configs/stage0.yaml
    'aggregation.strategy'.

    strategy == 'concatenate': join reviews per school (chronologically,
        respecting review_cutoff_year), and compute
        total_concatenated_word_count for later length-residualization at
        the school grain. No per-school minimum-review threshold is applied;
        n_reviews is retained as a covariate instead.
    strategy in {'mean','max','pct_mentioning'}: legacy per-review scoring
        + aggregation path (see Stage 3).
    """
    raise NotImplementedError


def build_school_table(cfg: dict) -> None:
    atlas = pd.read_csv(cfg["paths"]["raw_atlas"])
    reviews_raw = pd.read_csv(cfg["paths"]["raw_greatschools"])

    reviews_clean = clean_reviews(reviews_raw, cfg)
    reviews_clean.to_parquet(cfg["paths"]["out_reviews_clean"])

    review_agg = aggregate_reviews_to_school(reviews_clean, cfg)

    master = atlas.merge(review_agg, on="school_id", how="left")
    assert master["school_id"].is_unique, "Duplicate school_id in master table"

    match_rate = master["school_id"].notna().mean()
    if match_rate < cfg["qc"]["min_nces_match_rate"]:
        raise ValueError(f"NCES match rate {match_rate:.2%} below QC threshold")

    master.to_parquet(cfg["paths"]["out_schools_master"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    build_school_table(cfg)
