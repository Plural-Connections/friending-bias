"""
Theme-label helpers and balanced theme discovery.

GABRIEL phrases every label as "<subject> entries <verb> <theme> [more] than
<other> entries", where <subject> is the group hypothesised to have MORE of the
theme. The subject word (circle = low-FB, square = high-FB) plus the sign of the
measured signal together determine which group a theme is characteristic of.

Adapted from Nabeel's previous code for this pairwise discover/classify setup.
"""

import asyncio
import re
from pathlib import Path

import pandas as pd
import gabriel
from gabriel.utils import openai_utils as _gabriel_openai_utils

# GABRIEL's dynamic_timeout only raises its ceiling from *successful* call
# durations -- at full-corpus scale, if every call in a batch is too large to
# ever succeed, there's nothing to learn from and it retries at too low a
# timeout forever. Fixed 100-minute ceiling instead, so oversized calls get a
# real chance to finish rather than getting cancelled early.
MAX_TIMEOUT_SECONDS = 6000


def _reset_gabriel_client():
    """Drop GABRIEL's cached AsyncOpenAI client before starting a fresh
    asyncio.run() event loop.

    gabriel.utils.openai_utils caches one AsyncOpenAI client per process
    (module-level `_clients_async`, keyed only by base_url) and reuses it
    forever -- it never checks which event loop created it. Since every
    compare()/bucket() call here runs in its OWN asyncio.run() (a new loop
    each time), reusing that cached client after the first loop closes means
    its httpx connection-pool locks are bound to a now-dead loop: requests
    don't error, they just hang until get_all_responses's wall-clock timeout
    fires, then hang again on retry with the same broken client. No
    max_timeout value fixes this -- clearing the cache so a fresh client
    gets built on the current loop does.
    """
    _gabriel_openai_utils._clients_async.clear()


def theme_direction(label):
    """
    +1 if the label's SUBJECT is circle (low-FB), -1 if square (high-FB).
    Multiplying by net_pct then yields a signed "toward low-FB" score:
    positive => more characteristic of low-FB, negative => high-FB.
    """
    l = str(label).strip().lower()
    if l.startswith("circle"):
        return 1.0
    if l.startswith("square"):
        return -1.0
    return 1.0


def clean_theme_label(label):
    """Strip the '<grp> entries ... [more] than <grp> entries' scaffolding so the
    bar text is just the theme; the signed axis carries the group direction."""
    t = re.sub(r"(?i)^\s*(circle|square)\s+entries\s+", "", str(label))
    t = re.sub(r"(?i)\s+(more\s+)?than\s+(circle|square)\s+entries\.?\s*$", "", t)
    return t.strip()


def balanced_discover_themes(
    df_pairs,
    save_dir,
    model,
    instructions,
    bucket_count=10,
    use_dummy=False,
    reset_files=True,
):
    """
    Discover themes with GUARANTEED representation of BOTH groups.

    discover()'s single bucketing pass tends to fill every slot with the most
    prevalent/polarized themes, which (here) all favor high-FB schools even
    though ~44% of the raw candidate claims describe low-FB schools. To avoid
    that crowding, we run the pipeline's steps manually:
      1. compare() -> candidate distinguishing claims (each phrased
         "circle ... more than square" or "square ... more than circle")
      2. split candidates by subject (circle = low-FB, square = high-FB)
      3. bucket() EACH side separately into bucket_count//2 themes
      4. concatenate -> a balanced theme set
    Returns a bucket_df with columns ["bucket", "definition"].
    """
    sd = Path(save_dir)
    _reset_gabriel_client()
    cmp = asyncio.run(
        gabriel.compare(
            df_pairs,
            "low_fb_reviews",
            "high_fb_reviews",
            save_dir=str(sd / "compare"),
            model=model,
            differentiate=True,
            additional_instructions=instructions,
            reset_files=reset_files,
            use_dummy=use_dummy,
            dynamic_timeout=False,
            max_timeout=MAX_TIMEOUT_SECONDS,
        )
    )
    if cmp.empty or "attribute" not in cmp.columns:
        print(
            "compare() returned no parseable attribute/explanation pairs "
            f"(0 of {len(df_pairs)} pairs yielded valid JSON) — nothing to bucket."
        )
        return pd.DataFrame(columns=["bucket", "definition"])
    subj = cmp["attribute"].astype(str).str.strip().str.lower()
    per_side = max(1, bucket_count // 2)

    def _bucket_side(mask, name):
        sub = cmp[mask]
        term_defs = {}
        for a, e in zip(sub["attribute"], sub["explanation"]):
            if pd.notna(a) and a not in term_defs:
                term_defs[a] = str(e) if pd.notna(e) else ""
        if not term_defs:
            return pd.DataFrame(columns=["bucket", "definition"])
        cand = pd.DataFrame({"term": [term_defs]})  # discover's bucket input shape
        _reset_gabriel_client()
        return asyncio.run(
            gabriel.bucket(
                cand,
                "term",
                save_dir=str(sd / f"bucket_{name}"),
                model=model,
                bucket_count=per_side,
                differentiate=True,
                additional_instructions=instructions,
                reset_files=reset_files,
                use_dummy=use_dummy,
                dynamic_timeout=False,
                max_timeout=MAX_TIMEOUT_SECONDS,
            )
        )

    low_b = _bucket_side(subj.str.startswith("circle"), "low_fb")
    high_b = _bucket_side(subj.str.startswith("square"), "high_fb")
    print(
        f"balanced discovery: {len(low_b)} low-FB themes + {len(high_b)} high-FB themes"
    )
    return pd.concat([low_b, high_b], ignore_index=True)
