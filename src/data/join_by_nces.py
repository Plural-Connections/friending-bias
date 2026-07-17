"""Join the Social Capital Atlas onto GreatSchools demographics by NCES ID.

The two files name the same key differently:
  - social_capital_atlas.csv calls it "high_school"
  - gs_demos.csv           calls it "nces_id"

Both hold NCES school identifiers, but they are stored with different widths
(and some values lose leading zeros when a source treats them as integers).
We read both keys as strings and zero-pad them to the canonical 12-digit NCES
width before merging so that padding differences don't cause false misses.
"""

import pandas as pd

atlas_path = "data/raw/social_capital_atlas.csv"
demos_path = "data/raw/gs_demos.csv"
output_path = "data/interim/gs_demos_with_social_capital.csv"

NCES_WIDTH = 12

def normalize_nces(s):
    """Strip whitespace and zero-pad NCES ids to the canonical 12 digits."""
    return s.astype(str).str.strip().str.zfill(NCES_WIDTH)


def join_by_nces(atlas_path, demos_path, output_path, how="inner"):
    # dtype=str keeps leading zeros intact on both keys.
    atlas = pd.read_csv(atlas_path, dtype={"high_school": str})
    demos = pd.read_csv(demos_path, dtype={"nces_id": str})

    atlas["nces_id"] = normalize_nces(atlas["high_school"])
    demos["nces_id"] = normalize_nces(demos["nces_id"])

    merged = demos.merge(
        atlas.drop(columns="high_school"),
        on="nces_id",
        how=how,
    )

    n_atlas = atlas["nces_id"].nunique()
    n_demos = demos["nces_id"].nunique()
    n_matched = merged["nces_id"].nunique()
    print(f"Atlas schools: {n_atlas}, demos schools: {n_demos}")
    print(f"Matched schools: {n_matched}")

    # Report schools dropped by the inner join from each side.
    atlas_ids = set(atlas["nces_id"])
    demos_ids = set(demos["nces_id"])
    matched_ids = atlas_ids & demos_ids
    demos_dropped = n_demos - len(matched_ids)
    atlas_dropped = n_atlas - len(matched_ids)
    n_matched_ids = len(matched_ids)
    demos_rate = demos_dropped / n_demos if n_demos else 0
    atlas_rate = atlas_dropped / n_atlas if n_atlas else 0
    demos_match_rate = n_matched_ids / n_demos if n_demos else 0
    atlas_match_rate = n_matched_ids / n_atlas if n_atlas else 0
    print(f"Matched demos schools: {n_matched_ids} ({demos_match_rate:.1%})")
    print(f"Matched atlas schools: {n_matched_ids} ({atlas_match_rate:.1%})")
    print(f"Dropped demos schools: {demos_dropped} ({demos_rate:.1%})")
    print(f"Dropped atlas schools: {atlas_dropped} ({atlas_rate:.1%})")

    merged.to_csv(output_path, index=False)
    print(f"Wrote {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    join_by_nces(atlas_path, demos_path, output_path)
