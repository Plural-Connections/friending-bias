"""Topics-per-class comparison (BERTopic, nces-filtered corpus), reordered by
the size of the gap between each topic's high_fb and low_fb share -- rather
than by topic size/id order -- so the topics that differentiate the two
classes the most show up at the top of the chart, and the ones that
differentiate them least show up at the bottom.

Reads the raw frequency table already produced by
topics_BERT_classbased_filtered.py
(outputs/bertopic/class_based/topics_per_class_50_min_cluster_filtered.csv), re-derives
each topic's normalized Share of its class's reviews (the same
normalize_topic_frequencies() used for that script's existing normalized
chart), then sorts topics by |high_fb share - low_fb share| descending. This
is a different VIEW of an existing output, not a rescoring: nothing here
refits or reclassifies anything.

Topic labels are read directly from topic_summary_50_min_cluster_filtered
.csv's own keywords column rather than reloading the fitted BERTopic model,
since that's all a label needs here and it avoids a heavy model load for
what is otherwise a pure pandas/plotly reshaping script.

Outputs:
    outputs/bertopic/class_based/topics_per_class_50_min_cluster_filtered_gap_sorted.csv
        (topic_id, label, low_fb share, high_fb share, gap, rank)
    outputs/bertopic/class_based/topics_per_class_normalized_50_min_cluster_filtered_gap_sorted.png

Run from the project root as a module:
    python -m src.bertopic.class_based.topics_per_class_gap_sorted_filtered
"""

import os

import pandas as pd

from .topics_BERT_classbased import CLASS_DISPLAY_NAMES, normalize_topic_frequencies

table_path = "outputs/bertopic/class_based/topics_per_class_50_min_cluster_filtered.csv"
topic_summary_path = "outputs/bertopic/unsupervised_exploration/summary/topic_summary_50_min_cluster_filtered.csv"
output_table_path = "outputs/bertopic/class_based/topics_per_class_50_min_cluster_filtered_gap_sorted.csv"
output_fig_path = "outputs/bertopic/class_based/topics_per_class_normalized_50_min_cluster_filtered_gap_sorted.png"

NOISE_TOPIC = -1


def load_topic_labels(path=topic_summary_path, max_len=40):
    """topic_id -> truncated top-keywords label, straight from the fitted
    model's own c-TF-IDF keyword ranking (no need to reload the model)."""
    summary = pd.read_csv(path).set_index("topic_id")["keywords"]
    labels = {}
    for topic_id, kw in summary.items():
        label = ", ".join(str(kw).split(", ")[:3])
        labels[topic_id] = label[:max_len] + ("..." if len(label) > max_len else "")
    return labels


def build_gap_sorted_table(table_path=table_path):
    """One row per real topic (excludes noise), with each class's normalized
    Share and the absolute gap between them, sorted by gap descending."""
    known = pd.read_csv(table_path)
    known = known[known["Topic"] != NOISE_TOPIC].reset_index(drop=True)
    normalized = normalize_topic_frequencies(known)

    wide = normalized.pivot(index="Topic", columns="Class", values="Share").fillna(0.0)
    wide["gap"] = (wide["high_fb"] - wide["low_fb"]).abs()
    wide = wide.sort_values("gap", ascending=False).reset_index()
    wide = wide.rename(columns={"low_fb": "low_fb_share", "high_fb": "high_fb_share"})
    return wide


def plot_gap_sorted(wide, labels, title=None):
    """Grouped horizontal bar chart, ordered by gap (largest gap at top)."""
    import plotly.graph_objects as go

    topic_order = wide["Topic"].tolist()  # already gap-sorted, largest first
    label_map = {t: labels.get(t, f"topic_{t}") for t in topic_order}

    fig = go.Figure()
    for cls, share_col, color in [
        ("low_fb", "low_fb_share", "#56B4E9"),
        ("high_fb", "high_fb_share", "#E69F00"),
    ]:
        display_name = CLASS_DISPLAY_NAMES[cls]
        fig.add_trace(go.Bar(
            y=[label_map[t] for t in topic_order],
            x=wide[share_col],
            name=display_name,
            orientation="h",
            marker_color=color,
            hoverinfo="text",
            hovertext=[
                f"Topic {t}  ({label_map[t]})<br>{display_name} share: {s:.2f}%<br>gap: {g:.2f} pts"
                for t, s, g in zip(wide["Topic"], wide[share_col], wide["gap"])
            ],
        ))

    # Force exact top-to-bottom order (largest gap at top) rather than relying
    # on plotly's default category-order heuristics across two separate traces.
    label_order_bottom_to_top = [label_map[t] for t in reversed(topic_order)]
    fig.update_layout(
        barmode="group",
        title=title or "Topics per class, ranked by |High FB - Low FB| share gap (largest gap at top)",
        xaxis_title="% of class's reviews",
        yaxis_title="Topic",
        yaxis=dict(categoryorder="array", categoryarray=label_order_bottom_to_top),
        template="simple_white",
        width=1200,
        height=max(600, 45 * len(topic_order)),
        legend_title="Class",
    )
    return fig


def run():
    os.makedirs("outputs/bertopic/class_based", exist_ok=True)

    labels = load_topic_labels()
    wide = build_gap_sorted_table()
    wide["label"] = wide["Topic"].map(labels)
    wide.insert(0, "rank", range(1, len(wide) + 1))
    wide.to_csv(output_table_path, index=False)
    print(f"Wrote {output_table_path} ({len(wide)} topics, sorted by |high_fb - low_fb| share gap descending)")

    fig = plot_gap_sorted(wide, labels)
    fig.write_image(output_fig_path)
    print(f"Wrote {output_fig_path}")

    print("\n=== Top 10 topics by |high_fb - low_fb| share gap ===")
    with pd.option_context("display.max_colwidth", 40, "display.width", 140):
        cols = ["rank", "Topic", "label", "low_fb_share", "high_fb_share", "gap"]
        print(wide[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    run()
