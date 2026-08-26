"""Toy single-word sanity check for the topic-distribution pipeline: run a
handful of toy "reviews" through the already-fitted BERTopic model's
`.approximate_distribution` (the same method topics_distro_BERT_concat_filtered.py
and topics_distro_BERT_per_review.py use for real reviews) and inspect the
resulting per-topic share for each one -- a quick model-sanity spot check,
e.g. does a toy "review" that's just the word "bullying" actually land on
the topic whose keywords are about bullying?

Uses the CURRENT, most up-to-date model: MIN_CLUSTER_SIZE=50 on the
nces-filtered corpus, 85 real topics (see
topics_BERT_plain_unsupervised_filtered.py). This is a diagnostic, not a
real analysis -- the toy inputs are single words, not actual reviews, so
their distribution vectors are only meaningful as a "does this look right"
check, not as data to draw conclusions from.

Outputs:
    outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_50_min_cluster_filtered_toy_test.csv
        (one row per toy review, one column per real topic, holding that
        topic's approximate share of the toy review, plus top_topic /
        top_topic_share / top_topic_keywords convenience columns)

Run from the project root as a module:
    python -m src.eval.toy_test_topic_distro
"""

import os

import pandas as pd
from bertopic import BERTopic

MIN_CLUSTER_SIZE = 50
model_dir = f"outputs/models/bertopic/bertopic_model_{MIN_CLUSTER_SIZE}_min_cluster_filtered"
topic_summary_path = (
    f"outputs/bertopic/unsupervised_exploration/summary/topic_summary_{MIN_CLUSTER_SIZE}_min_cluster_filtered.csv"
)
output_path = (
    "outputs/bertopic/unsupervised_exploration/distributions/"
    f"topic_distributions_{MIN_CLUSTER_SIZE}_min_cluster_filtered_toy_test.csv"
)

# Same handful of single-word toy inputs as the original ad-hoc version of
# this check: two curriculum-tracking terms and two socially-loaded terms
# that should each land cleanly on their own topic, plus one deliberately
# generic word ("great") that should NOT land cleanly on any single topic --
# a review that's just "great" could be praising almost anything about a
# school, so a diffuse distribution here is the CORRECT/expected result, not
# a failure.
TOY_REVIEWS = ["ap", "ib", "bullying", "racism", "great"]

# Same window/stride as the real distribution scripts, for consistency --
# though with inputs this short, approximate_distribution falls back to
# using the whole toy "review" as a single token set regardless (window=8
# only matters once a document has at least 8 tokens).
WINDOW = 8
STRIDE = 4


def all_topic_ids(path=topic_summary_path):
    """Every real topic (excludes noise -1), sorted by size -- same
    convention as topics_distro_BERT_concat_filtered.py."""
    summary = pd.read_csv(path)
    summary = summary[summary["topic_id"] != -1].sort_values("size", ascending=False)
    return summary["topic_id"].tolist(), summary.set_index("topic_id")["keywords"]


def run():
    os.makedirs("outputs/bertopic/unsupervised_exploration/distributions", exist_ok=True)

    print(f"Loading fitted BERTopic model from {model_dir} ...")
    model = BERTopic.load(model_dir)

    topic_ids, keywords = all_topic_ids()
    print(f"Using all {len(topic_ids)} real topics (topic_id {min(topic_ids)}-{max(topic_ids)})")

    topic_distr, _ = model.approximate_distribution(
        TOY_REVIEWS,
        window=WINDOW,
        stride=STRIDE,
    )

    dist_df = pd.DataFrame(
        topic_distr[:, topic_ids],
        columns=[f"topic_{t}" for t in topic_ids],
    )
    dist_df.insert(0, "toy_review", TOY_REVIEWS)

    # Convenience columns so the "does this look right" check doesn't
    # require scanning all 85 columns by eye.
    topic_cols = [f"topic_{t}" for t in topic_ids]
    top_col = dist_df[topic_cols].idxmax(axis=1)
    dist_df["top_topic"] = top_col
    dist_df["top_topic_share"] = dist_df[topic_cols].max(axis=1)
    dist_df["top_topic_keywords"] = [keywords[int(c.split("_", 1)[1])] for c in top_col]

    dist_df.to_csv(output_path, index=False)
    print(f"\nWrote {output_path} ({len(TOY_REVIEWS)} toy reviews x {len(topic_ids)} topics)")

    print("\n=== Toy review -> top topic (sanity check) ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(
            dist_df[["toy_review", "top_topic", "top_topic_share", "top_topic_keywords"]]
            .to_string(index=False)
        )

    return dist_df


if __name__ == "__main__":
    run()
