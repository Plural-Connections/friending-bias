"""UNFILTERED, PER_REVIEW (fits on the unfiltered per-review corpus; the inner-join to
gs_demos_with_social_capital.csv below restricts the OUTPUT to schools with a
friending-bias score, but there is no separate _filtered.py sibling script):
Approximate per-school topic-mixture distributions, computed per REVIEW and
then aggregated -- an alternative to topics_distro_BERT_concat.py that avoids that
script's two aggregation problems:

  1. Boundary contamination: topics_distro_BERT_concat.py concatenates every review
     for a school into one blob (joined with a plain space, see
     concat_reviews.py) before calling .approximate_distribution(). BERTopic
     tokenizes that whole blob as a single stream and slides a window across
     it (see approximate_distribution's use of `self.vectorizer_model
     .build_tokenizer()` over the full document), so windows straddling a
     review boundary blend the tail of one review with the head of the next
     into a single nonsense "token set" that still gets scored against every
     topic. This gets worse the more reviews a school has -- worst for
     exactly the schools with the most data.
  2. Implicit word-count weighting: because the blob approach sums token-set
     similarities and L1-normalizes once per school, a single long review
     contributes proportionally more than several short ones, with no
     explicit decision behind it.

Calling .approximate_distribution() on each review INDIVIDUALLY sidesteps
both: no window ever crosses a review boundary (each doc is tokenized on its
own), and we control the aggregation explicitly -- here, a plain mean across
a school's reviews, so every review counts equally regardless of length.

Keyed on nces_id, not universal-id: the per-review file
(gs_reviews_sorted_by_day_posted.csv) carries an `nces_id` column, but it is
89% null there (only ~69K of 618K reviews have it filled in) -- nowhere near
reliable enough to key on directly. gs_demos_with_social_capital.csv has a
complete, 1:1 nces_id for every universal-id (0 nulls, no duplicates), the
same source concat_reviews.py already uses to attach nces_id. So we aggregate
by universal-id (always present on every review) and then inner-join in
nces_id from the demos file; the demos file is also the file
topic_regression_per_review.py ultimately joins against, so schools that fail this join were never going
to be usable downstream anyway.

Output: one row per school (nces_id, universal-id, n_reviews, n_words, and one
column per real topic), holding the mean of that topic's approximate
per-review share across the school's reviews.

Run from the project root as a module:
    python -m src.bertopic.unsupervised_exploration.topics_distro_BERT_per_review
"""

import os

import pandas as pd
from bertopic import BERTopic

reviews_path = "data/interim/gs_reviews_sorted_by_day_posted.csv"
demos_path = "data/interim/gs_demos_with_social_capital.csv"

TEXT_COL = "comments"
ID_COL = "universal-id"
NCES_ID_COL = "nces_id"

# Must match MIN_TOPIC_SIZE in topics_BERT_plain_unsupervised.py -- selects
# which fitted model/summary this run reads.
MIN_CLUSTER_SIZE = 200

model_dir = f"outputs/models/bertopic/bertopic_model_{MIN_CLUSTER_SIZE}_min_cluster"
topic_summary_path = f"outputs/bertopic/unsupervised_exploration/summary/topic_summary_{MIN_CLUSTER_SIZE}_min_cluster.csv"

# Earlier runs picked a hand-chosen TOP_N_TOPICS (40, then 300) out of a
# 952-topic model with a long fat tail, discarding most of a school's topical
# mass before the GBM ever saw it. At MIN_CLUSTER_SIZE=200 the model has only
# 235 real topics, few enough to use ALL of them as features -- no arbitrary
# cutoff needed, so none of that mass is discarded. `top_topic_ids(n=None)`
# below returns every real topic; the output filename records the count that
# turned out to be.

# Same window/stride as topics_distro_BERT_concat.py, kept for comparability. Reviews
# average ~82 words (max 768), so most reviews still exceed the window and get
# multiple token sets; short ones (<8 tokens) are used whole by BERTopic.
WINDOW = 8
STRIDE = 4

# Reviews are far shorter than the school-concatenated blobs (mean ~82 words
# vs. up to ~19,000), so each document produces far fewer token sets. A much
# larger batch than topics_distro_BERT_concat.py's 200 is safe here.
BATCH_SIZE = 2000


def load_reviews(path=reviews_path):
    """Read the per-review file and drop empty reviews (same filter as
    topics_BERT_plain_unsupervised.py, which fit the model on this corpus)."""
    df = pd.read_csv(path, usecols=[ID_COL, TEXT_COL])
    df = df.dropna(subset=[TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip().str.len() > 0]
    df["n_words"] = df[TEXT_COL].str.split().str.len()
    return df.reset_index(drop=True)


def top_topic_ids(path=topic_summary_path, n=None):
    """Reuse the model's own size ranking instead of re-deriving it a second
    way, so it always matches topic_summary_{MIN_CLUSTER_SIZE}_min_cluster.csv.
    n=None returns every real (non-noise) topic.
    """
    summary = pd.read_csv(path)
    summary = summary[summary["topic_id"] != -1].sort_values("size", ascending=False)
    top = summary if n is None else summary.head(n)
    return top["topic_id"].tolist(), top.set_index("topic_id")["keywords"]


def run():
    os.makedirs("outputs/bertopic/unsupervised_exploration/distributions", exist_ok=True)

    print(f"Loading fitted BERTopic model from {model_dir} ...")
    model = BERTopic.load(model_dir)

    reviews = load_reviews()
    docs = reviews[TEXT_COL].tolist()
    print(f"n reviews = {len(docs):,}  |  n schools (universal-id) = {reviews[ID_COL].nunique():,}")

    topic_ids, keywords = top_topic_ids(n=None)
    output_path = (
        f"outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_top{len(topic_ids)}"
        f"_min_cluster{MIN_CLUSTER_SIZE}_by_school_per_review.csv"
    )
    print(
        f"Using all {len(topic_ids)} real topics "
        f"(topic_id {min(topic_ids)}-{max(topic_ids)}) -- no top-N cutoff this run"
    )

    # Per review, not per school-concatenated blob: no window ever straddles
    # a review boundary. use_embedding_model=False and calculate_tokens=False
    # for the same reasons given in topics_distro_BERT_concat.py (fast c-TF-IDF
    # path; we only need the document-level table).
    topic_distr, _ = model.approximate_distribution(
        docs,
        window=WINDOW,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
    )

    # Columns are ordered by topic_id starting at 0 (outlier/-1 stripped
    # internally), so topic_ids can be used directly as column indices --
    # see the identical note in topics_distro_BERT_concat.py.
    per_review = pd.DataFrame(
        topic_distr[:, topic_ids],
        columns=[f"topic_{t}" for t in topic_ids],
    )
    per_review[ID_COL] = reviews[ID_COL].values
    per_review["n_words"] = reviews["n_words"].values

    topic_cols = [f"topic_{t}" for t in topic_ids]
    # Equal weight per review (plain mean), an explicit choice replacing the
    # implicit word-count weighting in topics_distro_BERT_concat.py's concatenated
    # approach.
    school = per_review.groupby(ID_COL, as_index=False).agg(
        n_reviews=(ID_COL, "size"),
        n_words=("n_words", "sum"),
        **{c: (c, "mean") for c in topic_cols},
    )

    demos = pd.read_csv(demos_path, dtype={NCES_ID_COL: str})[[ID_COL, NCES_ID_COL]]
    before = len(school)
    school = demos.merge(school, on=ID_COL, how="inner")
    print(
        f"\nJoined nces_id via {demos_path}: {len(school):,} / {before:,} "
        f"schools kept (schools absent from the demos/social-capital file are "
        f"dropped -- they aren't part of the modeling population either way)"
    )

    cols = [NCES_ID_COL, ID_COL, "n_reviews", "n_words"] + topic_cols
    school = school[cols]
    school.to_csv(output_path, index=False)

    print(f"\nWrote {output_path}  ({school.shape[0]:,} schools x {len(topic_ids)} topics)")
    print("\nTop-10 keyword reference for the columns above:")
    for t in topic_ids[:10]:
        print(f"  topic_{t}: {keywords[t]}")


if __name__ == "__main__":
    run()
