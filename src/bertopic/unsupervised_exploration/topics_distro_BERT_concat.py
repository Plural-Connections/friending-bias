"""UNFILTERED, CONCAT: Approximate per-school topic-mixture distributions over the fitted BERTopic
model (the model trained on individual reviews in
topics_BERT_plain_unsupervised.py).

Unit of analysis flips relative to that stage: SCHOOLS, not reviews. Each
school's `comments` field in gs_reviews_concat_by_school.csv concatenates
every review for that school into a single blob (up to ~900 reviews, ~19,000
words for the busiest schools), so it almost certainly mixes many topics
rather than expressing one dominant topic. Forcing a single hard-clustering
label onto that blob (.transform()) would throw away that mixture -- exactly
the failure mode BERTopic's docs call out:
    "each document is assigned to a single cluster and therefore also a
    single topic. In practice, documents may contain a mixture of topics...
    This is where .approximate_distribution comes in!"
It's also the documented path for scoring text the model never saw at fit
time -- these concatenated blobs were never part of the per-review training
corpus:
    "You can also approximate the topic distributions for unseen documents.
    It will not be as accurate as .transform but it is quite fast."

Output: one row per school, one column per top-TOP_N_TOPICS topic (by size,
reusing the ranking already reported in
outputs/bertopic/unsupervised_exploration/summary/topic_summary_50_min_cluster_top40.csv
so this table's "top 40" matches the topics already labeled elsewhere),
holding that topic's approximate share of the school's combined reviews.

This is the CONCAT-based aggregation method: every review is glued into one
document per school before scoring. It has two documented weaknesses relative
to its sibling, topics_distro_BERT_per_review.py (score each review
individually, then average): boundary contamination (BERTopic's scoring
window can straddle the seam between two different reviews in the glued
blob) and implicit word-count weighting (one long review can outweigh
several short ones in the final per-school mixture, since normalization
happens once per school rather than once per review). The output filename
carries a "_concat" suffix specifically so it's never confused with the
per-review method's output. It also still reads the oldest, unsuffixed
min_cluster=50 full-corpus model (outputs/models/bertopic/bertopic_model/) rather
than the current min_cluster=200 production model -- kept as-is so this
script continues to reproduce its original result rather than silently
changing scope; topic_regression_per_review.py uses the per-review method
against the production model instead.

Run from the project root as a module:
    python -m src.bertopic.unsupervised_exploration.topics_distro_BERT_concat
"""

import os

import pandas as pd
from bertopic import BERTopic

concat_path = "data/interim/gs_reviews_concat_by_school.csv"
model_dir = "outputs/models/bertopic/bertopic_model"
topic_summary_path = "outputs/bertopic/unsupervised_exploration/summary/topic_summary_50_min_cluster_top40.csv"
output_path = "outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_top40_by_school_concat.csv"

TEXT_COL = "comments"
ID_COL = "universal-id"
NCES_ID_COL = "nces_id"
N_REVIEWS_COL = "n_reviews"
N_WORDS_COL = "n_words"

# Matches TOP_N_TOPICS in topics_BERT_plain_unsupervised.py, so "top 40" means
# the same 40 topics everywhere in the project.
TOP_N_TOPICS = 40

# window/stride: taken directly from the docs' own guidance for a corpus this
# size. They suggest "this window between 4 and 8" in general, and for a
# "very large dataset" specifically recommend "stride=4 and window=8 ...
# [which] increases the computational speed quite a bit." Our unit here --
# 92,902 school-concatenated blobs, some tens of thousands of words long --
# is exactly that large-dataset case, so we use their large-dataset numbers
# rather than a small default stride=1.
WINDOW = 8
STRIDE = 4

# batch_size: exists so that "Creating token sets for each document" doesn't
# "result in quite a large list of token sets" that "might not fit into
# memory anymore ... we can process batches of documents instead to minimize
# the memory overload." The docs' own example uses batch_size=500, but that
# example is written for ordinary single-paragraph documents (20 Newsgroups
# posts); our documents concatenate up to ~900 individual reviews each, so a
# batch of 500 could still bundle together several outlier-long schools at
# once. We use a smaller batch so the same memory guard stays meaningful for
# our much heavier documents.
BATCH_SIZE = 200


def load_school_docs(path=concat_path):
    """Read the school-concatenated review file and drop empty blobs."""
    df = pd.read_csv(
        path,
        usecols=[ID_COL, NCES_ID_COL, TEXT_COL, N_REVIEWS_COL, N_WORDS_COL],
        dtype={NCES_ID_COL: str},
    )
    df = df.dropna(subset=[TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip().str.len() > 0]
    return df.reset_index(drop=True)


def top_topic_ids(path=topic_summary_path, n=TOP_N_TOPICS):
    """Reuse the model's own size ranking instead of re-deriving "top 40" a
    second way, so it always matches topic_summary_50_min_cluster_top40.csv.
    """
    summary = pd.read_csv(path)
    summary = summary[summary["topic_id"] != -1].sort_values("size", ascending=False)
    top = summary.head(n)
    return top["topic_id"].tolist(), top.set_index("topic_id")["keywords"]


def run():
    os.makedirs("outputs/bertopic/unsupervised_exploration/distributions", exist_ok=True)

    print(f"Loading fitted BERTopic model from {model_dir} ...")
    model = BERTopic.load(model_dir)

    meta = load_school_docs()
    docs = meta[TEXT_COL].tolist()
    print(f"n schools = {len(docs):,}")

    topic_ids, keywords = top_topic_ids()
    print(
        f"Restricting to top {len(topic_ids)} topics by size "
        f"(topic_id {min(topic_ids)}-{max(topic_ids)})"
    )

    # use_embedding_model=False (the default): "we compare the c-TF-IDF
    # calculations between the token sets and all topics. Due to its
    # bag-of-word representation, this is quite fast. However, you might want
    # to use the ... embedding_model instead ... [but] it is often
    # computationally quite a bit slower" -- with ~93K long documents we keep
    # the fast c-TF-IDF path, consistent with how this model already
    # describes its topics (c-TF-IDF keywords, not embedding similarity).
    #
    # calculate_tokens=False (the default): token-level distributions are
    # "computation-wise more expensive and can require more memory" and exist
    # only "for visualization purposes in
    # topic_model.visualize_approximate_distribution" -- we only need the
    # document-level table, so we skip it.
    topic_distr, _ = model.approximate_distribution(
        docs,
        window=WINDOW,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
    )

    # approximate_distribution's columns are ordered by topic_id starting at
    # 0 (BERTopic strips the outlier/-1 column internally before returning
    # this matrix -- see `self.c_tf_idf_[self._outliers:]` in its source), so
    # topic_distr[:, k] is topic k's share for every document; topic_ids can
    # be used directly as column indices.
    dist_df = pd.DataFrame(
        topic_distr[:, topic_ids],
        columns=[f"topic_{t}" for t in topic_ids],
    )

    out = pd.concat(
        [meta[[ID_COL, NCES_ID_COL, N_REVIEWS_COL, N_WORDS_COL]], dist_df],
        axis=1,
    )
    out.to_csv(output_path, index=False)

    print(f"\nWrote {output_path}  ({out.shape[0]:,} schools x {len(topic_ids)} topics)")
    print("\nTop-10 keyword reference for the columns above:")
    for t in topic_ids[:10]:
        print(f"  topic_{t}: {keywords[t]}")


if __name__ == "__main__":
    run()
