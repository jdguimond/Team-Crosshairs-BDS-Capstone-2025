#!/usr/bin/env python3
import pandas as pd
import glob
import os

results_dir = "results/identified_off_targets"
output_csv = "results/identified_off_targets/collected_summary.csv"

files = glob.glob(os.path.join(results_dir, "*_identified_matched.txt"))

all_dfs = []

for f in files:
    try:
        df = pd.read_csv(f, sep="\t", low_memory=False)

        cols = [
            "Name",
            "WindowSequence",
            "Site_SubstitutionsOnly.Sequence",
            "Cell",
            "Targetsite",
            "TargetSequence",
            "Position.Pvalue",
            "Narrow.Pvalue",
            "Position.Control.Pvalue",
            "Narrow.Control.Pvalue"
        ]

        sub = df[cols].copy()

        accession = os.path.basename(f).split("_")[0]
        sub.insert(0, "Accession", accession)

        all_dfs.append(sub)

    except Exception as e:
        print(f"[warn] Skipping {f}: {e}")

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"[info] Wrote {len(final_df)} rows into {output_csv}")
else:
    print("[warn] No files processed.")
