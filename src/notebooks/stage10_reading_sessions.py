"""Stage 10 — qualitative validation pass.

Pulls schools with largest positive/negative Stage 6 residuals-of-residuals
and schools with extreme scores on top concepts, for manual reading.
No modeling code — this automates *selection*, not the reading itself.
"""

def select_schools_to_read(residuals_df, concept_matrix, top_n: int = 10):
    raise NotImplementedError
