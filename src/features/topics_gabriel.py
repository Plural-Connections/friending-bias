"""
Held-out contrastive theme discovery (the package entry point).

Themes are DISCOVERED on one half of the school pairs and MEASURED on the other,
so the reported effect sizes are out-of-sample — free of the in-sample
("garden of forking paths") inflation you get when the same data both proposes
and evaluates a theme.
"""

import asyncio
from pathlib import Path

import numpy as np
import gabriel
from dotenv import load_dotenv

from .length_adjust import length_adjusted_pairwise
from .pairs import build_combined_labels, build_fb_pairs, summarize_actual_inverted
from .plotting import plot_discovered_themes, plot_prevalence_themes
from .themes import MAX_TIMEOUT_SECONDS, _reset_gabriel_client, balanced_discover_themes

load_dotenv(Path(__file__).parent / ".env")  # API key

# All reviews for one U.S. high school are concatenated into a single text; this
# tells GABRIEL what the corpora are and steers it toward substance over style.
INSTRUCTIONS = (
    "Each text is ALL of the GreatSchools reviews for a single U.S. high "
    "school, concatenated together (parent/student/teacher reviews mixed). "
    "Identify what differentiates these two groups of SCHOOLS based on what is "
    "actually happening inside the building and in the community around it — "
    "concrete practices, policies, incidents, staff behavior, programs, "
    "facilities, and outcomes that reviewers describe.\n\n"
    "Do NOT propose themes about how the reviews are WRITTEN: ignore tone, "
    "sentiment intensity, formality, enthusiasm, politeness, gratitude, "
    "grammar, review length, vocabulary, or reviewer style/identity. A theme "
    "like 'circle entries express more gratitude than square entries' or "
    "'group A entries are written more informally' is NOT acceptable — it "
    "describes the writing, not the school. A theme like 'group A entries "
    "describe more responsive communication from teachers' or 'group B entries "
    "describe more frequent bullying incidents' IS acceptable — it describes a "
    "substantive, actionable characteristic of the school itself.\n\n"
    "Also avoid themes built on proper nouns, school names, locations, or other "
    "identifying details specific to one school rather than a generalizable "
    "pattern across schools.\n\n"
    "MERGE, but only when two candidate themes restate the SAME underlying "
    "fact: before finalizing your theme list, check every pair against this "
    "test — 'if I deleted one of these, would the other one now just be "
    "saying the same thing in different words?' Only merge when the answer "
    "is yes. Do NOT merge just because the same reviews tend to mention both "
    "— reviews routinely praise or criticize several different things about "
    "a school in one breath, so co-occurring in the same text is NOT "
    "evidence that two themes are one construct. Do not split a topic into "
    "narrower sub-themes (depth vs. breadth, frequency vs. severity, one "
    "specific example vs. the general pattern) unless the two sub-themes "
    "would be supported by clearly different, non-overlapping evidence.\n\n"
    "Watch for one recurring failure mode in particular: folding a concrete, "
    "specific offering or practice (a particular kind of course, program, "
    "policy, or facility the school does or doesn't have) into a broad, "
    "holistic quality-of-experience theme (general teaching quality, "
    "engagement, morale, climate) just because schools praised for the "
    "specific offering are often praised generally too. What a school "
    "concretely provides and how the delivery/experience feels are "
    "DIFFERENT facts — keep them as separate themes even when the same "
    "review touches both. For example:\n"
    "  - 'more advanced/honors-level course offerings' and 'more rigorous "
    "coursework' -> merge into one 'academic program strength' theme (same "
    "underlying fact), but do NOT further merge that into a broader "
    "'teaching quality' or 'instructional experience' theme — course "
    "availability/rigor and how well classes are taught are different "
    "facts, even if the same review praises both.\n"
    "  - 'more bullying incidents' and 'less effective response to bullying' "
    "-> merge into one 'bullying/school safety climate' theme.\n"
    "  - 'unresponsive teachers' and 'unresponsive administration' -> merge "
    "into one 'poor staff communication/responsiveness' theme, unless the "
    "review evidence clearly distinguishes teacher-level from admin-level "
    "communication.\n"
    "  - 'outdated facilities' and 'crowded classrooms' -> these ARE distinct "
    "(different evidence: building condition vs. enrollment/space), so keep "
    "separate.\n"
    "When you are unsure whether two candidate themes restate the same "
    "fact, default to keeping them separate, not merging — a theme list "
    "where every entry still names a specific, concrete practice is "
    "strongly preferred over a smaller set of vaguer, catch-all themes. If "
    "a merged definition becomes so broad it no longer names a specific "
    "practice, policy, or offering (e.g. a vague 'instructional experience' "
    "or 'overall quality' label), that is a sign of over-merging — split it "
    "back into the more specific themes it absorbed. Each final theme "
    "should be describable in one sentence that could NOT also describe any "
    "other theme in the set."
)


def discover_contrasting_themes_heldout(
    input_file_demos="data/interim/gs_demos_with_social_capital.csv",
    input_file_reviews_concat="data/interim/gs_reviews_concat_by_school.csv",
    save_dir="outputs/gabriel_discover_heldout",
    model="gpt-5-mini",
    max_schools_per_group=None,
    split_frac=0.5,
    seed=0,
    balance_directions=True,
    bucket_count=10,
    max_words_per_school=None,
    reset_files=True,
    use_dummy=False,
):
    """
    Splits the low/high school PAIRS into two disjoint halves, then:
      * DISCOVER themes on split A only (gabriel.discover)
      * MEASURE those fixed themes on the held-out split B (gabriel.classify),
        computing net_pct + confidence intervals on B.

    Because the themes are chosen on A and scored on B, the reported effect
    sizes are out-of-sample.

    `use_dummy=True` runs the whole pipeline with stubbed model responses (no API
    calls / cost) to smoke-test plumbing. Requires OPENAI_API_KEY otherwise
    (GABRIEL reads it from the environment).
    """
    # seed randomizes the within-pair low<->high correspondence too.
    df_pairs = build_fb_pairs(
        input_file_demos,
        input_file_reviews_concat,
        max_schools_per_group,
        seed=seed,
        max_words_per_school=max_words_per_school,
    )

    # Deterministic shuffle + disjoint split into A (discover) / B (measure).
    # Offset the seed so the A/B partition is independent of the pairing shuffle.
    order = np.random.default_rng(seed + 1).permutation(len(df_pairs))
    n_a = int(round(len(df_pairs) * split_frac))
    df_a = df_pairs.iloc[order[:n_a]].reset_index(drop=True)
    df_b = df_pairs.iloc[order[n_a:]].reset_index(drop=True)
    print(
        f"pairs: {len(df_pairs)} total -> discover A: {len(df_a)} | measure B: {len(df_b)}"
    )

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) DISCOVER themes on split A only ──────────────────────────────────
    if balance_directions:
        # Bucket each direction separately so BOTH low-FB and high-FB themes are
        # represented (plain discover tends to fill every slot with high-FB).
        bucket_df = balanced_discover_themes(
            df_a,
            save_dir=str(out_dir / "discover_A"),
            model=model,
            instructions=INSTRUCTIONS,
            bucket_count=bucket_count,
            reset_files=reset_files,
            use_dummy=use_dummy,
        )
    else:
        res_a = asyncio.run(
            gabriel.discover(
                df_a,
                circle_column_name="low_fb_reviews",
                square_column_name="high_fb_reviews",
                save_dir=str(out_dir / "discover_A"),
                model=model,
                additional_instructions=INSTRUCTIONS,
                bucket_count=bucket_count,
                reset_files=reset_files,
                use_dummy=use_dummy,
            )
        )
        bucket_df = res_a.get("bucket_df")
    if bucket_df is None or not len(bucket_df):
        print("No themes discovered on split A; aborting.")
        return {}
    base_labels = dict(zip(bucket_df["bucket"], bucket_df["definition"]))
    print(f"discovered {len(base_labels)} themes on split A")

    # ── 2) MEASURE those fixed themes on held-out split B ───────────────────
    combined, rename = build_combined_labels(base_labels)
    _reset_gabriel_client()
    clf_b = asyncio.run(
        gabriel.classify(
            df_b,
            labels=combined,
            differentiate=True,
            circle_column_name="low_fb_reviews",
            square_column_name="high_fb_reviews",
            save_dir=str(out_dir / "classify_B"),
            model=model,
            additional_instructions=INSTRUCTIONS,
            reset_files=reset_files,
            use_dummy=use_dummy,
            dynamic_timeout=False,
            max_timeout=MAX_TIMEOUT_SECONDS,
        )
    )
    clf_b = clf_b.rename(columns=rename)
    summary_b = summarize_actual_inverted(clf_b, base_labels)
    if not len(summary_b):
        print(
            "WARNING: summary is empty — none of the discovered theme labels "
            "matched the classification columns. This usually means a stale "
            "cache was reused; re-run with reset_files=True (default) or clear "
            f"{out_dir}/classify_B."
        )

    # ── Persist + plot (CIs are now out-of-sample) ──────────────────────────
    bucket_df.to_csv(out_dir / "themes_from_A.csv", index=False)
    summary_b.to_csv(out_dir / "summary_heldoutB.csv", index=False)
    clf_b.to_csv(out_dir / "classification_heldoutB.csv", index=False)
    print(f"Saved held-out outputs to {out_dir}/")

    if len(summary_b):
        plot_discovered_themes(
            summary_b,
            save_path=out_dir / "distinctive_themes_heldout.png",
            classification=clf_b,
        )

    # Length-adjusted companion chart (controls for within-pair review length).
    adjusted = length_adjusted_pairwise(clf_b)
    if len(adjusted):
        adjusted.to_csv(
            out_dir / "distinctive_themes_heldout_adjusted.csv", index=False
        )
        plot_prevalence_themes(
            adjusted,
            save_path=out_dir / "distinctive_themes_heldout_adjusted.png",
            title="Pairwise net difference, ADJUSTED for within-pair review length",
            y_label="<- more in HIGH-FB     net diff at equal length (pts)     more in LOW-FB ->",
        )

    return {
        "themes": bucket_df,
        "summary_B": summary_b,
        "classification_B": clf_b,
        "adjusted_B": adjusted,
    }


if __name__ == "__main__":
    discover_contrasting_themes_heldout()
