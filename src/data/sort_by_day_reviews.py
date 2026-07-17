"""Sort all reviews in a CSV by chronological order.

The "posted" column holds ISO 8601 timestamps, e.g. 2007-05-05T00:00:00-07:00.
Because the format is fixed-width (YYYY-MM-DDT...), a plain string sort orders
them correctly at the day level, and a plain string comparison against a year
prefix filters them correctly at the year level.
"""

import pandas as pd

input_path = "data/raw/gs_reviews.csv"
output_path = "data/interim/gs_reviews_sorted_by_day_posted.csv"

# Keep only reviews posted in this year or earlier (drops 2023 and later).
THROUGH_YEAR = 2022

def sort_reviews(input_path, output_path, column="posted", through_year=THROUGH_YEAR):
    df = pd.read_csv(input_path)
    before = len(df)
    df = df[df[column].str[:4].astype(int) <= through_year]
    print(f"Dropped {before - len(df)} rows posted after {through_year}")
    df = df.sort_values(column, kind="stable")
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows sorted by {column!r} to {output_path}")


if __name__ == "__main__":
    sort_reviews(input_path, output_path)
