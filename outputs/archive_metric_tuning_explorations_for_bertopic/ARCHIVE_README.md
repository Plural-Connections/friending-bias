# Metric-tuning explorations for BERTopic (archive)

This folder collects BERTopic-related outputs that no longer correspond to
any current script's write-path in `src/` -- earlier specifications tried
and superseded on the way to the current pipeline. Nothing here is read or
written by any script as of this writing (verified by grepping `src/` for
every path listed below); it is kept purely as a record of what was tried,
in what order, and why each step was abandoned, so a later reader isn't
left guessing what these files are or worrying they're a missing dependency.

The one related artifact NOT moved here is
`outputs/models/bertopic/bertopic_model/` (the original, unsuffixed
min_cluster=50 full-corpus model) -- despite looking like it belongs in
this history, it is still a live, actively-read dependency of
`src/bertopic/unsupervised_exploration/topics_distro_BERT_concat.py` (moved
from `src/features/topics_distro_BERT.py` in the later src/ reorg, then
renamed again to add the `_concat` suffix), so it stays where it is.

Ordered roughly by when each was produced (file mtimes), reconstructed after
the fact from filenames, timestamps, and `reports/stage3_concept_validation.md`:

## 1. `tables/single_BERT_topics/` -- earliest exploration (Jul 22 - Aug 4)

- `topic_summary_0.01_reviews_100_min_cluster.csv`,
  `topic_assignments_0.01_reviews_100_min_cluster..csv` (Jul 22),
  `topic_representative_docs_0.01_reviews_100_min_cluster..csv` (Jul 22) --
  and the `0.005_reviews_50_min_cluster` equivalents (Jul 30). The naming
  suggests two hyperparameters were being explored jointly at this point (a
  minimum-review-fraction threshold alongside a minimum cluster size),
  before the pipeline consolidated to the single `MIN_TOPIC_SIZE` knob used
  today in `topics_BERT_plain_unsupervised.py`. The stray double-dot in the
  `100_min_cluster..csv` filenames is a leftover artifact from whatever
  script produced them, not a typo introduced later -- left as-is rather
  than "fixed," since renaming an archived file would misrepresent what was
  actually produced.
- `topic_hierarchy_min_cluster50_original.csv`,
  `topic_tree_min_cluster50_original.txt` (Aug 3) -- output of
  `src/features/topics_BERT_hierarchical.py` (Stage 2b, at that path back
  then -- now `src/bertopic/hierarchical_exploration/topics_BERT_hierarchical.py`),
  run against the
  original min_cluster=50 full-corpus model. Renamed from the bare
  `topic_hierarchy.csv`/`topic_tree.txt` they were originally written as,
  since the current script (now against the production min_cluster=200
  model) writes the exact same bare filenames to
  `outputs/bertopic/hierarchical_exploration/` -- the collision was
  otherwise easy to mistake for a duplicated file rather than two distinct
  models' outputs. Re-running the current script will *not* touch these
  archived copies either way, since it writes to that other directory.
- `topic_assignments_50_min_cluster_top40.csv`,
  `topic_representative_docs_50_min_cluster_top40.csv`,
  `topic_representative_docs_formatted_50_min_cluster_top40.md` (Aug 4) --
  an earlier naming convention for `topics_BERT_plain_unsupervised.py`'s
  output, before the "_top40" suffix was dropped from the assignments/
  representative-docs filenames (the current version's outputs are named
  `topic_assignments_{MIN_TOPIC_SIZE}_min_cluster.csv`, with no topic-count
  suffix, since it now exports all topics rather than a fixed top-N slice).
  The `.md` file is a hand-formatted version of the representative-docs
  table for easier reading, not something any script generates.
- `topics_per_class_min_cluster50_original.csv` (Aug 4 13:55) -- output of
  `src/features/topics_BERT_classbased.py` (at that path back then -- now
  `src/bertopic/class_based/topics_BERT_classbased.py`), from before that script's
  output paths were changed to embed `_{MIN_CLUSTER_SIZE}_min_cluster` (see
  `figures/topics_per_class_min_cluster50_original.png` below for its
  paired chart). Renamed from the bare `topics_per_class.csv` it was
  originally written as, for the same collision reason as the hierarchy
  files above.

## 2. `figures/topics_per_class_min_cluster50_original.png`, `figures/topics_per_class_normalized_min_cluster50_original.png` (Aug 4 13:55)

Paired with `tables/single_BERT_topics/topics_per_class_min_cluster50_original.csv`
above -- the same pre-naming-fix run of `topics_BERT_classbased.py`, against
the original min_cluster=50 full-corpus model. Renamed from the bare
`topics_per_class.png`/`topics_per_class_normalized.png` they were
originally written as. The current, actively-maintained versions are
`outputs/bertopic/class_based/topics_per_class_200_min_cluster.png` (full
corpus, current production min_cluster=200 model) and
`outputs/bertopic/class_based/topics_per_class_50_min_cluster_filtered.png`
(nces-matched corpus, min_cluster=50 model).

## 3. The R^2 debugging trail: concat vs. per-review vs. topic-count (Aug 6 - Aug 12)

This is the specification search described in the paper's "Why
Review-Content R^2 Stayed Low" section and in
`reports/notes_on_concept_validation_and_gabriel_instruction_log.md`. All of
it predates the eventual structural fix (refitting at min_cluster=200 so all
235 topics could be retained with no cutoff), which is what produced the
current, live
`outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_top235_min_cluster200_by_school_per_review.csv`
and the `topic_regression_per_review.py` results -- neither of which is
archived here.

| File(s) | Date | R^2 (gbm) | What it tested |
|---|---|---|---|
| `tables/topic_gbm_metrics_concat.csv`, `topic_gbm_importances_concat.csv` | Aug 6 | **0.0103** | Concatenation aggregation (glue all of a school's reviews into one blob, score once), top-40 topics, min_cluster=50. The method later found to have boundary-contamination and implicit word-count-weighting problems -- see `topics_distro_BERT_concat.py`'s docstring. |
| `tables/topic_distributions_top40_by_school_per_review.csv`, `topic_gbm_metrics_per_review.csv`, `topic_gbm_importances_per_review.csv`, `models/topic_gbm.pkl` | Aug 12 (~18:04-18:11) | **0.0145** | Fixed the aggregation method (score each review individually, average per school) but kept the top-40-of-952-topics feature cutoff. R^2 moved by only ~0.4 percentage points (0.0103 -> 0.0145) -- confirming the aggregation bug, while real, was not the dominant driver of the low R^2. |
| `tables/topic_distributions_top300_by_school_per_review.csv`, `topic_gbm_metrics_top300.csv`, `topic_gbm_importances_top300.csv`, `models/topic_gbm_top300.pkl` | Aug 12 (~18:39-18:49) | **0.0300** | Widened the retained feature set from the top 40 to the top 300 (of 952) topics, per-review aggregation, still min_cluster=50. Confirmed the real bottleneck was topic-space coverage (top 40 topics captured only 44.9% of non-noise review mass), not the aggregation method -- but only recovered R^2 to ~3%. |
| *(not archived -- current production file)* `outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_top235_min_cluster200_by_school_per_review.csv` | Aug 12 23:48 | **0.035** | The structural fix: refit BERTopic entirely at min_cluster=200 (235 real topics) so ALL topics could be retained as features, no cutoff at all, rather than further widening an arbitrary one. This is the number reported in the paper as the full-corpus production result. |

The takeaway carried into the paper: R^2 barely moved across this entire
search (0.010 -> 0.015 -> 0.030 -> 0.035), which is itself evidence that the
low content-only R^2 reflects a real ceiling in the data (corroborated
separately by a raw bivariate-correlation check and a synthetic-signal
recovery test, both described in the paper), not a specification artifact
that further tuning would have resolved.
