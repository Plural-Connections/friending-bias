"""Topics per class (BERTopic, nces-filtered corpus), splitting the flat
topics from topics_BERT_plain_unsupervised_filtered.py by friending-bias
group.

Same technique as topics_BERT_classbased.py -- BERTopic's "Topics per Class"
doesn't refit anything: it takes the topics already discovered over the WHOLE
(filtered) corpus and, for each topic, recomputes a *local* c-TF-IDF
representation restricted to just the documents in one class. See:
https://maartengr.github.io/BERTopic/getting_started/topicsperclass/topicsperclass.html

Class here = the reviewed school's friending-bias (FB) group, same
circle/square convention as src/gabriel/pairs.py: a school is "low_fb" if
bias_own_ses_hs < 0, "high_fb" if > 0.

Unlike the unfiltered script, `unknown_fb` should end up nearly empty here:
every review in gs_reviews_sorted_by_day_posted_filtered.csv already has a
matching nces_id (and therefore a bias score, since
gs_demos_with_social_capital.csv only contains schools with one) -- filtering
happened upstream in sort_by_day_reviews_nces_filtered.py, not in this script.
It's kept only for the (bias_own_ses_hs == 0) edge case and general
robustness, mirroring the unfiltered script's bookkeeping requirement that
topics_per_class() needs exactly one class per document.

We reuse the ALREADY-fitted MIN_CLUSTER_SIZE=50 filtered model (outputs/models/
bertopic/bertopic_model_50_min_cluster_filtered/, fit in
topics_BERT_plain_unsupervised_filtered.py on the 69,238-review filtered
corpus, 85 real topics) when present -- no re-embedding/re-clustering needed.
MIN_CLUSTER_SIZE is hardcoded here (not imported from
topics_BERT_plain_unsupervised_filtered.py's mutable MIN_TOPIC_SIZE) for the
same reason given in topics_BERT_classbased.py: decoupling this script from
that module's state so it can't silently point at a deleted/different model.

Outputs:
    outputs/bertopic/class_based/topics_per_class_50_min_cluster_filtered.csv
    outputs/bertopic/class_based/topics_per_class_50_min_cluster_filtered.png
    outputs/bertopic/class_based/topics_per_class_normalized_50_min_cluster_filtered.png
"""

import os

import pandas as pd

from .topics_BERT_classbased import normalize_topic_frequencies, plot_topics_per_class
from ..unsupervised_exploration.topics_BERT_plain_unsupervised_filtered import (
    NCES_ID_COL,
    TEXT_COL,
    fit_topics,
    load_reviews,
)

MIN_CLUSTER_SIZE = 50  # matches the current filtered model
model_dir = f"outputs/models/bertopic/bertopic_model_{MIN_CLUSTER_SIZE}_min_cluster_filtered"
topic_summary_path = f"outputs/bertopic/unsupervised_exploration/summary/topic_summary_{MIN_CLUSTER_SIZE}_min_cluster_filtered.csv"

demos_path = "data/interim/gs_demos_with_social_capital.csv"
table_path = f"outputs/bertopic/class_based/topics_per_class_{MIN_CLUSTER_SIZE}_min_cluster_filtered.csv"
fig_path = f"outputs/bertopic/class_based/topics_per_class_{MIN_CLUSTER_SIZE}_min_cluster_filtered.png"
fig_path_normalized = (
    f"outputs/bertopic/class_based/topics_per_class_normalized_{MIN_CLUSTER_SIZE}_min_cluster_filtered.png"
)

TOP_N_TOPICS = None  # None = every real (non-noise) topic, currently 85
UNKNOWN_CLASS = "unknown_fb"


def load_or_fit_base_model():
    """Reuse the saved filtered flat-topic model + its docs if available
    (fast: no re-embedding/re-clustering needed), else fit fresh on the
    filtered corpus."""
    from bertopic import BERTopic

    meta = load_reviews()
    docs = meta[TEXT_COL].tolist()

    try:
        model = BERTopic.load(model_dir)
        if len(model.topics_) != len(docs):
            raise ValueError(
                f"saved model has {len(model.topics_)} docs, "
                f"but load_reviews() now returns {len(docs)} — stale model, refitting"
            )
        print(f"Loaded existing model from {model_dir}/ ({len(docs):,} docs)")
    except Exception as exc:
        print(f"Could not reuse saved model ({exc}); fitting fresh on the filtered corpus.")
        model, _, _ = fit_topics(docs, min_topic_size=MIN_CLUSTER_SIZE)

    return model, docs, meta


def assign_fb_classes(meta, demos_path=demos_path):
    """One class per review, positionally aligned with `meta`'s row order:
    'low_fb', 'high_fb', or UNKNOWN_CLASS if the review's school has no Atlas
    friending-bias score. Keyed on nces_id (see topics_BERT_classbased.py)."""
    demos = pd.read_csv(demos_path, dtype={NCES_ID_COL: str})[
        [NCES_ID_COL, "bias_own_ses_hs"]
    ]
    bias_by_school = demos.set_index(NCES_ID_COL)["bias_own_ses_hs"]

    bias = meta[NCES_ID_COL].map(bias_by_school)

    classes = pd.Series(UNKNOWN_CLASS, index=meta.index)
    classes[bias < 0] = "low_fb"
    classes[bias > 0] = "high_fb"
    return classes.tolist()


def top_topic_ids(path=topic_summary_path, n=TOP_N_TOPICS):
    """Topic ids ranked by the filtered model's OWN `size` column (i.e. how
    many filtered-corpus reviews landed in that topic at fit time), read
    straight from topic_summary_{MIN_CLUSTER_SIZE}_min_cluster_filtered.csv --
    not recomputed via model.get_topic_freq() global-frequency ranking (which
    would mix in noise from however `known` happens to be filtered/grouped
    here). Excludes the noise topic -1. n=None (the default) returns every
    real topic."""
    summary = pd.read_csv(path)
    summary = summary[summary["topic_id"] != -1].sort_values("size", ascending=False)
    top = summary if n is None else summary.head(n)
    return top["topic_id"].tolist()


def run():
    os.makedirs("outputs/bertopic/class_based", exist_ok=True)

    model, docs, meta = load_or_fit_base_model()
    classes = assign_fb_classes(meta)

    n_low = classes.count("low_fb")
    n_high = classes.count("high_fb")
    n_unknown = classes.count(UNKNOWN_CLASS)
    print(f"n reviews = {len(docs):,}  |  low_fb = {n_low:,}  |  "
          f"high_fb = {n_high:,}  |  unknown_fb (excluded) = {n_unknown:,}\n")

    topics_per_class = model.topics_per_class(docs, classes=classes)

    known = topics_per_class[topics_per_class["Class"] != UNKNOWN_CLASS].reset_index(drop=True)
    known.to_csv(table_path, index=False)
    print(f"Wrote {table_path} ({len(known)} topic x class rows)")

    # Source the plotted topic list directly from the filtered model's own
    # topic_summary file (ranked by its `size` column), not from a
    # global-frequency recomputation -- see top_topic_ids(). TOP_N_TOPICS=None
    # means every real topic, so n_plot below is the model's actual topic count.
    plot_topics = top_topic_ids()
    n_plot = len(plot_topics)
    print(f"Plotting all {n_plot} topics sourced from {topic_summary_path}")

    fig = plot_topics_per_class(model, known, top_n_topics=n_plot, top_topics=plot_topics)
    fig.write_image(fig_path)
    print(f"Wrote {fig_path}")

    normalized = normalize_topic_frequencies(known)
    fig_norm = plot_topics_per_class(
        model, normalized, top_n_topics=n_plot, top_topics=plot_topics,
        value_col="Share", xaxis_title="% of class's reviews",
        title=f"All {n_plot} topics per class (filtered, min cluster size {MIN_CLUSTER_SIZE}): "
              f"Low FB vs High FB (normalized by class review count)",
    )
    fig_norm.write_image(fig_path_normalized)
    print(f"Wrote {fig_path_normalized}")

    print(f"\n=== All {n_plot} topics per class (low_fb vs high_fb), by frequency ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        top = (
            known.sort_values(["Class", "Frequency"], ascending=[True, False])
            .groupby("Class")
            .head(n_plot)
        )
        print(top.to_string(index=False))

    return known


if __name__ == "__main__":
    run()
