"""UNFILTERED, no filtered counterpart by design: Hierarchical topic
structure (BERTopic), on top of the flat topics from
topics_BERT_plain_unsupervised.py (the UNFILTERED full-corpus model).

This stage is exploratory/archival -- a one-off look at how the flat topics
nest, not a result any later stage builds on -- so unlike the other
topics_BERT_*.py scripts it was deliberately never given an nces-filtered
counterpart. Keep it pointed at the unfiltered model rather than adding a
_filtered.py sibling.

The plain/flat pass gives ~950 independent topics with no notion of which ones
are near-duplicates or minor variants of a broader theme. This stage adds that
structure: BERTopic's `hierarchical_topics()` computes the cosine distance
between every pair of topics' c-TF-IDF vectors, then runs hierarchical
(agglomerative) clustering over those distances — the same technique described
here:
https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html#linkage-functions

This does NOT change any topic or its assignments — it only adds a merge tree
on top of the topics that already exist, useful for:
  * spotting near-duplicate topics worth merging before treating the flat
    topic list as final
  * picking a natural "collapse level" (how many topics to keep) instead of
    treating all ~950 flat topics as equally distinct
  * distinguishing a genuinely standalone topic (its nearest neighbor in the
    tree is still a large distance away) from one that's just a fragment of a
    bigger theme (merges in almost immediately)

Linkage function: BERTopic's default is Ward linkage (minimizes within-cluster
variance at each merge — good for compact, evenly-sized clusters). The doc page
above lists the scipy alternatives — single, complete, average, centroid,
median — each with a different merging bias (e.g. single-linkage chains
together anything with even one close pair, complete-linkage only merges
clusters that are close on their *farthest* members). LINKAGE_METHOD below is
a plain string knob so a different one can be tried without touching the
pipeline logic.

We reuse the ALREADY-fitted MIN_CLUSTER_SIZE=200 production model
(outputs/models/bertopic/bertopic_model_200_min_cluster/) when present, since
hierarchical_topics() only needs the fitted topics + each topic's c-TF-IDF
vector, not the raw embeddings — no need to re-embed/re-cluster the full
corpus. Falls back to a fresh fit (also at min_topic_size=200) if no saved
model exists yet.

MIN_CLUSTER_SIZE is hardcoded here rather than imported from
topics_BERT_plain_unsupervised.py's mutable MIN_TOPIC_SIZE, for the same
reason given in topics_BERT_classbased.py: that module's value has drifted
across experiments before (e.g. to 400, later reverted), and importing it
directly would let this script silently point at a different/nonexistent
model whenever someone is mid-experiment with that constant.

Outputs:
    outputs/bertopic/hierarchical_exploration/topic_hierarchy.csv  (hierarchical_topics() merge tree)
    outputs/bertopic/hierarchical_exploration/topic_tree.txt       (human-readable text tree)
    outputs/bertopic/hierarchical_exploration/topic_hierarchy.png (static dendrogram image -- NOT an
                                         interactive HTML page, despite what
                                         BERTopic's own visualize_hierarchy()
                                         is normally used for; write_image()
                                         here rasterizes it because a static
                                         image is easier to embed elsewhere)

Run from the project root as a module:
    python -m src.bertopic.hierarchical_exploration.topics_BERT_hierarchical
"""

import os

from scipy.cluster import hierarchy as sch

from ..class_based.topics_BERT_classbased import clean_topic_label
from ..unsupervised_exploration.topics_BERT_plain_unsupervised import (
    NOISE_TOPIC,
    TEXT_COL,
    fit_topics,
    load_reviews,
)

MIN_CLUSTER_SIZE = 200  # matches the production full-corpus model everywhere else
model_dir = f"outputs/models/bertopic/bertopic_model_{MIN_CLUSTER_SIZE}_min_cluster"

hierarchy_path = "outputs/bertopic/hierarchical_exploration/topic_hierarchy.csv"
tree_path = "outputs/bertopic/hierarchical_exploration/topic_tree.txt"
hierarchy_fig_path = "outputs/bertopic/hierarchical_exploration/topic_hierarchy.png"

# Ward is BERTopic's own default. Alternatives from the linked "Linkage
# functions" section: "single", "complete", "average", "centroid", "median".
LINKAGE_METHOD = "complete"


def load_or_fit_base_model(reviews_path_arg=None):
    """Reuse the saved flat-topic model + its docs if available (fast: no
    re-embedding/re-clustering needed), else fit fresh on the full corpus."""
    from bertopic import BERTopic

    meta = load_reviews(reviews_path_arg) if reviews_path_arg else load_reviews()
    docs = meta[TEXT_COL].tolist()

    try:
        model = BERTopic.load(model_dir)
        if len(model.topics_) != len(docs):
            raise ValueError(
                f"saved model has {len(model.topics_)} docs, "
                f"but load_reviews() now returns {len(docs)} — stale model, refitting"
            )
        topics = model.topics_
        print(f"Loaded existing model from {model_dir}/ ({len(docs):,} docs)")
    except Exception as exc:
        print(f"Could not reuse saved model ({exc}); fitting fresh on the full corpus.")
        model, topics, _ = fit_topics(docs, min_topic_size=MIN_CLUSTER_SIZE)

    return model, docs, topics, meta


def build_hierarchy(model, docs, linkage_method=LINKAGE_METHOD):
    """Ward (or other scipy linkage) hierarchy over topics' c-TF-IDF distances."""
    linkage_function = lambda x: sch.linkage(x, linkage_method, optimal_ordering=True)
    return model.hierarchical_topics(docs, linkage_function=linkage_function)


def run():
    os.makedirs("outputs/bertopic/hierarchical_exploration", exist_ok=True)

    model, docs, topics, meta = load_or_fit_base_model()
    n_topics = len(set(topics) - {NOISE_TOPIC})
    print(f"n reviews = {len(docs):,}  |  n flat topics (excl. noise) = {n_topics}\n")

    hier = build_hierarchy(model, docs)
    hier.to_csv(hierarchy_path, index=False)
    print(f"Wrote {hierarchy_path} ({len(hier)} merges, linkage={LINKAGE_METHOD!r})")

    tree_text = model.get_topic_tree(hier)
    with open(tree_path, "w") as f:
        f.write(tree_text)
    print(f"Wrote {tree_path}")

    # Replace BERTopic's default underscore-joined labels (e.g.
    # "4_love school_love_best school") with clean_topic_label()'s
    # "Topic 4: love school, love, best school" before rendering.
    model.set_topic_labels({tid: clean_topic_label(model, tid, max_words=3, max_len=40)
                             for tid in set(topics)})

    fig = model.visualize_hierarchy(
        hierarchical_topics=hier, width=1400, height=max(600, 18 * n_topics),
        custom_labels=True,
    )
    fig.write_image(hierarchy_fig_path)
    print(f"Wrote {hierarchy_fig_path}")

    print("\n=== Topic hierarchy (text tree) ===")
    print(tree_text)

    return hier


if __name__ == "__main__":
    run()
