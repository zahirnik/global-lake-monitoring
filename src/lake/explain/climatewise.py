"""
Per-climate-class explainability analysis.

Fixes two bugs that existed in the legacy `src/climatewise.py`:
1. The legacy `__main__` accidentally called `lake_wise_analysis(...)` instead
   of `climate_wise_analysis(...)` — so it actually produced lake-wise
   results regardless of file name.
2. Outputs landed in `./EXPs/lakewise/` instead of a climate-specific folder.

The clean version below groups the dataframe by climate code, then re-uses
the lake-wise analyzer on each climate's subset, writing one set of CSVs per
climate class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from lake.explain.lakewise import lake_wise_analysis


def climate_wise_analysis(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    climate_column: str = "Climate",
) -> Dict[str, Dict[str, Path]]:
    """Run the lake-wise analyzer separately for each climate class.

    Parameters
    ----------
    dataframe
        The Lake_Density master dataset.
    output_dir
        Parent directory. One sub-folder is created per climate class.
    climate_column
        Column to group by (defaults to "Climate"; some datasets call this
        column "Koppen" or similar — keep the option exposed).

    Returns
    -------
    Mapping of climate-class -> {csv-kind: path} as produced by
    :func:`lake_wise_analysis`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Path]] = {}
    for climate, sub_df in dataframe.groupby(climate_column, sort=True):
        # `groupby` may yield mixed-type group keys; coerce to a safe folder name.
        safe_name = str(climate).replace("/", "_").replace(" ", "_") or "Unknown"
        climate_dir = output_dir / safe_name
        print(f"[climate={safe_name}] {len(sub_df)} rows, "
              f"{sub_df['ID'].nunique()} lakes")
        results[safe_name] = lake_wise_analysis(sub_df, output_dir=climate_dir)

    return results
