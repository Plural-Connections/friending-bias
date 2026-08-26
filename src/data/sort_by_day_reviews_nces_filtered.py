"""Sort all reviews in a CSV by chronological order.

Same as sort_by_day_reviews.py, but drops any review whose "nces_id" doesn't
match gs_demos_with_social_capital.csv (and therefore has no friending-bias
score) instead of keeping it with a null nces_id.

The "posted" column holds ISO 8601 timestamps, e.g. 2007-05-05T00:00:00-07:00.
Because the format is fixed-width (YYYY-MM-DDT...), a plain string sort orders
them correctly at the day level, and a plain string comparison against a year
prefix filters them correctly at the year level.

Drops reviews with blank/missing "comments" here, at the source, rather than
leaving each downstream consumer to filter them out itself (see
sort_by_day_reviews.py for the full rationale).
"""

import pandas as pd

input_path = "data/raw/gs_reviews.csv"
demos_path = "data/interim/gs_demos_with_social_capital.csv"
output_path = "data/processed/gs_reviews_sorted_by_day_posted_filtered.csv"

# Keep only reviews posted in this year or earlier (drops 2023 and later).
THROUGH_YEAR = 2022

def sort_reviews(input_path, output_path, demos_path=demos_path,
                 column="posted", through_year=THROUGH_YEAR, group="universal-id",
                 text_column="comments"):
    df = pd.read_csv(input_path)
    before = len(df)
    df = df[df[column].str[:4].astype(int) <= through_year]
    print(f"Dropped {before - len(df)} rows posted after {through_year}")

    before = len(df)
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].astype(str).str.strip().str.len() > 0]
    print(f"Dropped {before - len(df)} rows with blank/missing {text_column!r}")

    df = df.sort_values(column, kind="stable")

    nces_by_school = pd.read_csv(demos_path)[[group, "nces_id"]]
    before = len(df)
    df = df.merge(nces_by_school, on=group, how="inner")
    print(f"Dropped {before - len(df)} rows with no matching nces_id")

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows sorted by {column!r} to {output_path}")


if __name__ == "__main__":
    sort_reviews(input_path, output_path)
