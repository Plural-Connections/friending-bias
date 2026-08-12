"""Mean review word count and review count by friending-bias (FB) group,
at the school level.

Uses the same low_fb / high_fb convention as topics_BERT_classbased.py
(mirroring the GABRIEL pipeline's circle/square convention in
src/features/pairs.py): a school is "low_fb" if bias_own_ses_hs < 0,
"high_fb" if > 0. Schools with no Atlas bias score get "unknown_fb" and are
dropped before reporting the mean.

Word counts come from gs_reviews_concat_by_school.csv's "n_words" column --
one row per school, the word count of that school's reviews concatenated
together (see concat_reviews.py) -- not per-review word counts.

Review counts come from that same file's "n_reviews" column rather than
being re-derived here from gs_reviews_sorted_by_day_posted.csv: concat_reviews.py
already drops blank/missing-comment rows before counting, so its "n_reviews"
only counts reviews that actually have text. Re-deriving counts from the raw
sorted file separately would double-count that filtering step and, if it
missed the same blank-comment filter, disagree with every other reader of
this data (see sort_by_day_reviews.py, which now drops blank comments at the
source precisely so nothing downstream needs to filter them again).

Keyed on "nces_id" rather than "universal-id": gs_reviews_concat_by_school.csv
carries nces_id directly (joined in concat_reviews.py from
gs_demos_with_social_capital.csv), so both sides of the join here use it.
Rows with no nces_id (school wasn't in the NCES join, see concat_reviews.py
and sort_by_day_reviews.py) are dropped before assigning a class.

"""

import pandas as pd

demos_path = "data/interim/gs_demos_with_social_capital.csv"
reviews_path = "data/interim/gs_reviews_concat_by_school.csv"

UNKNOWN_CLASS = "unknown_fb"


def assign_fb_classes(schools, bias_by_school):
    """One class per school, positionally aligned with `schools`'s row order."""
    bias = schools["nces_id"].map(bias_by_school)
    classes = pd.Series(UNKNOWN_CLASS, index=schools.index)
    classes[bias < 0] = "low_fb"
    classes[bias > 0] = "high_fb"
    return classes


def run(demos_path=demos_path, reviews_path=reviews_path):
    demos = pd.read_csv(demos_path, dtype={"nces_id": str})[
        ["nces_id", "bias_own_ses_hs"]
    ]
    bias_by_school = demos.set_index("nces_id")["bias_own_ses_hs"]

    schools = pd.read_csv(reviews_path, dtype={"nces_id": str})[
        ["nces_id", "n_words", "n_reviews"]
    ].dropna(subset=["nces_id"])

    schools["Class"] = assign_fb_classes(schools, bias_by_school)

    known = schools[schools["Class"] != UNKNOWN_CLASS]
    summary = (
        known.groupby("Class")[["n_words", "n_reviews"]]
        .agg(["mean", "count"])
    )
    summary.columns = ["mean_wordcount", "n_schools", "mean_reviewcount", "_n_schools2"]
    summary = summary.drop(columns="_n_schools2").reset_index()

    total_reviews = known.groupby("Class")["n_reviews"].sum().rename("total_reviews")
    summary = summary.merge(total_reviews, on="Class")
    print(summary.to_string(index=False))

    low_mean_words = summary.loc[summary["Class"] == "low_fb", "mean_wordcount"].item()
    high_mean_words = summary.loc[summary["Class"] == "high_fb", "mean_wordcount"].item()
    pct_larger_words = (high_mean_words / low_mean_words - 1) * 100
    print(f"high_fb class has a {pct_larger_words:.1f}% larger mean wordcount")

    low_mean_reviews = summary.loc[summary["Class"] == "low_fb", "mean_reviewcount"].item()
    high_mean_reviews = summary.loc[summary["Class"] == "high_fb", "mean_reviewcount"].item()
    pct_larger_reviews = (high_mean_reviews / low_mean_reviews - 1) * 100
    print(f"high_fb class has a {pct_larger_reviews:.1f}% larger mean review count")

    low_total_reviews = summary.loc[summary["Class"] == "low_fb", "total_reviews"].item()
    high_total_reviews = summary.loc[summary["Class"] == "high_fb", "total_reviews"].item()
    pct_larger_total_reviews = (high_total_reviews / low_total_reviews - 1) * 100
    print(f"high_fb class has {pct_larger_total_reviews:.1f}% more total reviews")

    return summary


if __name__ == "__main__":
    run()
