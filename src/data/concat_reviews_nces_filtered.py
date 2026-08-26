"""Concatenate all reviews from the same school in chronological order.

Same as concat_reviews.py, but keeps only schools that have an nces_id —
i.e. schools that matched gs_demos_with_social_capital.csv (and therefore
also have a friending-bias score, since that file already drops schools
without one).

The input CSV is already sorted chronologically by "posted" (see
sort_by_day_reviews.py). A stable group-by on "universal-id" therefore
preserves that chronological order within each school, so joining the
"comments" of each group yields one concatenated review per school,
oldest first.
"""

import pandas as pd

input_path = "data/interim/gs_reviews_sorted_by_day_posted.csv"
demos_path = "data/interim/gs_demos_with_social_capital.csv"
output_path = "data/processed/gs_reviews_concat_by_school_nces_filtered.csv"

SEPARATOR = " "

def concat_reviews(input_path, output_path, demos_path=demos_path,
                   group="universal-id", text="comments", sep=SEPARATOR):
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

    nces_by_school = pd.read_csv(demos_path)[[group, "nces_id"]]
    grouped = grouped.merge(nces_by_school, on=group, how="inner")

    grouped.to_csv(output_path, index=False)
    print(f"Wrote {len(grouped)} schools to {output_path}")


if __name__ == "__main__":
    concat_reviews(input_path, output_path)
