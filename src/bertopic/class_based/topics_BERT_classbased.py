"""UNFILTERED: Topics per class (BERTopic), splitting the flat topics from
topics_BERT_plain_unsupervised.py (the UNFILTERED full-corpus model) by
friending-bias group -- see topics_BERT_classbased_filtered.py for the
counterpart built on the nces-filtered model.

BERTopic's "Topics per Class" technique doesn't refit anything: it takes the
topics already discovered over the WHOLE corpus and, for each topic, recomputes
a *local* c-TF-IDF representation restricted to just the documents in one class
— answering "how is this topic talked about within group X" rather than
finding new topics per group. See:
https://maartengr.github.io/BERTopic/getting_started/topicsperclass/topicsperclass.html

Class here = the reviewed school's friending-bias (FB) group, mirroring the
same circle/square convention the GABRIEL pipeline uses (src/gabriel/pairs.py):
a school is "low_fb" if bias_own_ses_hs < 0, "high_fb" if > 0. Reviews whose
school has no Atlas bias score (most of the corpus — only ~14K of ~93K schools
were matched, see reports/notes_on_data_processing.md) get a bookkeeping class of
"unknown_fb" and are dropped before export/plotting. They can't just be left
out upfront: topics_per_class() requires exactly one class per document,
positionally aligned with the fitted model's docs — no gaps allowed.

This is a NO-API, zero-marginal-cost complement to the GABRIEL contrastive
pipeline (src/gabriel/topics_gabriel.py): it asks a version of the same
question — which themes are disproportionately common in low- vs high-FB
schools — using topics that are already sitting in the saved model, with no
risk of the timeout/cost problems that pipeline hit.

We reuse the ALREADY-fitted MIN_CLUSTER_SIZE=200 model (outputs/models/bertopic/
bertopic_model_200_min_cluster/, the same one topic_regression_per_review.py and
topics_distro_BERT_per_review.py already treat as the production full-corpus
model, 235 real topics -- see topics_BERT_plain_unsupervised.py's docstring
for why 200 replaced the original 50) when present, same fast-path pattern as
topics_BERT_hierarchical.py — no re-embedding/re-clustering needed.

MIN_CLUSTER_SIZE is hardcoded here rather than imported from
topics_BERT_plain_unsupervised.py's mutable MIN_TOPIC_SIZE: that module's
value has changed across experiments (e.g. to 400, whose model was later
deleted), so importing it made this script silently point at whatever
config someone last left that file in -- including a model_dir that no
longer exists on disk, in which case load_or_fit_base_model() would fall
back to refitting from scratch on the full ~618K-review corpus rather than
reusing the intended production model. Hardcoding the number here decouples
this script from that module's state.

Outputs:
    outputs/bertopic/class_based/topics_per_class_200_min_cluster.csv (topic x class rows: words, frequency)
    outputs/bertopic/class_based/topics_per_class_200_min_cluster.png    (grouped bar chart, raw frequency)
    outputs/bertopic/class_based/topics_per_class_normalized_200_min_cluster.png
        (same chart, but each topic's Frequency is expressed as a % of that
        class's total review count instead of a raw count. high_fb has more
        reviews than low_fb overall -- not just per-school as in
        high_low_fb_wordcount_reviewcount.py's 13.7% mean-reviews-per-school
        figure, but in total documents fed to this topic model too (see
        normalize_topic_frequencies()) -- so raw frequency alone makes every
        topic look more common in high_fb even when it's equally or less
        prevalent as a SHARE of that class's reviews. Dividing by each
        class's own total puts both classes on the same (%-of-class) scale.

"""

import os

import pandas as pd

from ..unsupervised_exploration.topics_BERT_plain_unsupervised import (
    NCES_ID_COL,
    TEXT_COL,
    fit_topics,
    load_reviews,
)

MIN_CLUSTER_SIZE = 200  # matches the production full-corpus model everywhere else
model_dir = f"outputs/models/bertopic/bertopic_model_{MIN_CLUSTER_SIZE}_min_cluster"

demos_path = "data/interim/gs_demos_with_social_capital.csv"
table_path = f"outputs/bertopic/class_based/topics_per_class_{MIN_CLUSTER_SIZE}_min_cluster.csv"
fig_path = f"outputs/bertopic/class_based/topics_per_class_{MIN_CLUSTER_SIZE}_min_cluster.png"
fig_path_normalized = f"outputs/bertopic/class_based/topics_per_class_normalized_{MIN_CLUSTER_SIZE}_min_cluster.png"

TOP_N_TOPICS = 40
UNKNOWN_CLASS = "unknown_fb"

# Display text for the internal "low_fb"/"high_fb" class values -- kept
# separate from those values themselves, which stay as-is since they're also
# used to index/filter the underlying data.
CLASS_DISPLAY_NAMES = {"low_fb": "Low FB", "high_fb": "High FB"}


def clean_topic_label(model, topic_id, max_words=4, max_len=45, include_id=True):
    """Human-readable topic label built straight from the model's own top
    keywords, e.g. "Topic 4: love school, love, best school" -- instead of
    BERTopic's default `topic_labels_` representation, which underscore-joins
    the topic id and every keyword (e.g. "4_love school_love_best school").

    include_id=False drops the "Topic N: " prefix, keeping just the
    comma-separated keywords -- used for the topics-per-class charts, where
    the topic number isn't meaningful to a reader (matches the label style
    of the gap-sorted chart)."""
    words = [w for w, _ in model.get_topic(topic_id)[:max_words]]
    prefix = f"Topic {topic_id}: " if include_id else ""
    label = prefix + ", ".join(words)
    return label if len(label) <= max_len else label[: max_len - 3] + "..."


def load_or_fit_base_model():
    """Reuse the saved flat-topic model + its docs if available (fast: no
    re-embedding/re-clustering needed), else fit fresh on the full corpus."""
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
        print(f"Could not reuse saved model ({exc}); fitting fresh on the full corpus.")
        model, _, _ = fit_topics(docs, min_topic_size=MIN_CLUSTER_SIZE)

    return model, docs, meta


def assign_fb_classes(meta, demos_path=demos_path):
    """One class per review, positionally aligned with `meta`'s row order:
    'low_fb', 'high_fb', or UNKNOWN_CLASS if the review's school has no Atlas
    friending-bias score.

    Keyed on nces_id rather than universal-id (matching
    high_low_fb_wordcount_reviewcount.py's convention): some distinct
    universal-ids resolve to the same physical school's nces_id, so joining
    on universal-id can split one school's reviews across two labels. Reviews
    with a null nces_id (school wasn't in the NCES join, see
    sort_by_day_reviews.py) fall through to UNKNOWN_CLASS along with schools
    that have no Atlas bias score.
    """
    demos = pd.read_csv(demos_path, dtype={NCES_ID_COL: str})[
        [NCES_ID_COL, "bias_own_ses_hs"]
    ]
    bias_by_school = demos.set_index(NCES_ID_COL)["bias_own_ses_hs"]

    bias = meta[NCES_ID_COL].map(bias_by_school)

    classes = pd.Series(UNKNOWN_CLASS, index=meta.index)
    classes[bias < 0] = "low_fb"
    classes[bias > 0] = "high_fb"
    return classes.tolist()


def normalize_topic_frequencies(known):
    """Add a "Share" column: each row's Frequency as a % of that row's Class's
    total Frequency across all topics (including the -1 outlier topic, so
    shares are relative to the whole class, not just the plotted top-N).

    high_fb has more total reviews than low_fb in this corpus too, not just
    the mean-reviews-per-school gap in high_low_fb_wordcount_reviewcount.py
    (13.7%) -- so raw Frequency systematically runs higher for high_fb across
    almost every topic. Dividing by each class's own total puts both classes
    on a common %-of-class-reviews scale, which is what's actually comparable.
    """
    known = known.copy()
    class_totals = known.groupby("Class")["Frequency"].transform("sum")
    known["Share"] = known["Frequency"] / class_totals * 100
    return known


def plot_topics_per_class(model, known, top_n_topics=TOP_N_TOPICS,
                           value_col="Frequency", xaxis_title="Frequency",
                           title=None, top_topics=None):
    """Grouped horizontal bar chart: for each of the top-N topics, one bar for
    low_fb and one for high_fb -- 2 * top_n_topics bars total, all visible at
    once.

    `top_topics`: an explicit, already-ordered list of topic ids to plot. If
    omitted, falls back to the top-N topics by GLOBAL frequency (summed
    across both classes, via model.get_topic_freq()) -- the original
    behavior, kept as the default for callers that don't pass their own list.
    Pass an explicit list to source the topic ordering from somewhere else
    (e.g. a topic_summary_*.csv file's own `size` ranking) instead.

    BERTopic's own `visualize_topics_per_class()` puts one TOPIC per trace
    (not one class per trace) and only sets the first topic's trace `visible`
    -- every other topic is `visible="legendonly"`, meant to be toggled on by
    clicking its legend entry in an interactive HTML page. `fig.write_image()`
    respects that same visibility setting, so a static PNG export only ever
    shows the first topic's 2 bars; the other topics are computed but simply
    never drawn. This builds the same data as a plain grouped bar chart
    instead, with every bar visible, since a static image has no clicking.

    The y-axis label must be the same string for a topic in BOTH classes, or
    plotly can't recognize the two classes' bars as belonging to the same
    category to group them side by side. `known["Words"]` is computed LOCALLY
    per class (that's the whole point of topics-per-class), so it can differ
    slightly between low_fb and high_fb for the same topic -- using it in the
    label is what caused bars to render as two disjoint blocks instead of
    grouped pairs. The model's global `topic_labels_` is identical regardless
    of class, so that's the grouping key; per-class `Words` still shows up in
    the hover text, just not in the label itself.
    """
    import plotly.graph_objects as go

    if top_topics is None:
        freq_df = model.get_topic_freq()
        freq_df = freq_df.loc[freq_df.Topic != -1, :]
        top_topics = freq_df.sort_values("Count", ascending=False)["Topic"].head(top_n_topics).tolist()

    topic_names = {topic: clean_topic_label(model, topic, include_id=False) for topic in model.topic_labels_}

    data = known[known["Topic"].isin(top_topics)].copy()
    order = {topic: i for i, topic in enumerate(top_topics)}
    data["_order"] = data["Topic"].map(order)
    data = data.sort_values("_order", ascending=False)  # most frequent at the top
    data["Label"] = data["Topic"].map(topic_names)

    fig = go.Figure()
    for cls, color in [("low_fb", "#56B4E9"), ("high_fb", "#E69F00")]:
        sub = data[data["Class"] == cls]
        display_name = CLASS_DISPLAY_NAMES[cls]
        fig.add_trace(go.Bar(
            y=sub["Label"], x=sub[value_col], name=display_name,
            orientation="h", marker_color=color,
            hoverinfo="text",
            hovertext=[
                f"Topic {t}<br>Words ({display_name}): {w}<br>{value_col}: {v:.3g}"
                for t, w, v in zip(sub["Topic"], sub["Words"], sub[value_col])
            ],
        ))

    # Force the exact top-to-bottom order rather than relying on plotly's
    # default category-order heuristics across two separate traces.
    label_order_bottom_to_top = [topic_names[t] for t in reversed(top_topics)]

    fig.update_layout(
        barmode="group",
        title=title or f"Top {top_n_topics} topics by global frequency: Low FB vs High FB",
        xaxis_title=xaxis_title,
        yaxis_title="Topic",
        yaxis=dict(categoryorder="array", categoryarray=label_order_bottom_to_top),
        template="simple_white",
        width=1200,
        height=max(600, 45 * top_n_topics),
        legend_title="Class",
    )
    return fig


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

    # Drop the unknown-FB bucket before exporting/plotting -- it exists only
    # because topics_per_class() requires a class for every document, not
    # because it's a group worth comparing.
    known = topics_per_class[topics_per_class["Class"] != UNKNOWN_CLASS].reset_index(drop=True)
    known.to_csv(table_path, index=False)
    print(f"Wrote {table_path} ({len(known)} topic x class rows)")

    fig = plot_topics_per_class(model, known, top_n_topics=TOP_N_TOPICS)
    fig.write_image(fig_path)
    print(f"Wrote {fig_path}")

    normalized = normalize_topic_frequencies(known)
    fig_norm = plot_topics_per_class(
        model, normalized, top_n_topics=TOP_N_TOPICS,
        value_col="Share", xaxis_title="% of class's reviews",
        title=f"Top {TOP_N_TOPICS} topics by global frequency: Low FB vs High FB "
              f"(normalized by class review count)",
    )
    fig_norm.write_image(fig_path_normalized)
    print(f"Wrote {fig_path_normalized}")

    print(f"\n=== Top {TOP_N_TOPICS} topics per class (low_fb vs high_fb), by frequency ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        top = (
            known.sort_values(["Class", "Frequency"], ascending=[True, False])
            .groupby("Class")
            .head(TOP_N_TOPICS)
        )
        print(top.to_string(index=False))

    return known


if __name__ == "__main__":
    run()
