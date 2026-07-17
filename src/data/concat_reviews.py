"""Concatenate all reviews from the same school in chronological order.

The input CSV is already sorted chronologically by "posted" (see
sort_by_day_review.py). A stable group-by on "universal-id" therefore
preserves that chronological order within each school, so joining the
"comments" of each group yields one concatenated review per school,
oldest first.
"""

import pandas as pd

input_path = "data/interim/gs_reviews_sorted_by_day_posted.csv"
output_path = "data/interim/gs_reviews_concat_by_school.csv"

SEPARATOR = " "

def concat_reviews(input_path, output_path, group="universal-id",
                   text="comments", sep=SEPARATOR):
    df = pd.read_csv(input_path)
    # Drop rows with no review text so they don't leak in as "nan" or
    # inflate n_reviews.
    df = df.dropna(subset=[text])
    grouped = (
        df.groupby(group, sort=False)[text]
        .apply(lambda s: sep.join(s.astype(str)))
        .reset_index()
    )
    grouped["n_reviews"] = (
        df.groupby(group, sort=False)[text].size().values
    )
    grouped["n_words"] = grouped[text].str.split().str.len()
    grouped.to_csv(output_path, index=False)
    print(f"Wrote {len(grouped)} schools to {output_path}")


if __name__ == "__main__":
    concat_reviews(input_path, output_path)
