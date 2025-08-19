import os
import pandas as pd
from Levenshtein import distance as levenshtein

CIRCLESEQ_CSV = "results/identified_off_targets/collected_summary.csv"
ZEBRAFISH_CSV = "ZebrafishPaper_45sgRNAs_targetSequences_2025-7-20.csv"
OUTPUT_DIR = "results/identified_off_targets/model_inputs"


def load_data():
    """Load both Circle-seq and Zebrafish CSVs."""
    cs = pd.read_csv(CIRCLESEQ_CSV)
    zb = pd.read_csv(ZEBRAFISH_CSV)

    # normalize column names
    cs = cs.rename(columns={
        "Site_SubstitutionsOnly.Sequence": "OffTargetSequence",
        "TargetSequence": "TargetSequence"
    })

    zb = zb.rename(columns={
        "off_target_sequence": "OffTargetSequence",
        "sgRNA_target_sequence": "TargetSequence"
    })

    return cs, zb


def merge_datasets(cs, zb, max_dist=2):
    """
    Merge Circle-seq (TRUE cleavage) and Zebrafish (default FALSE cleavage),
    using approximate matching between off-target sequences.
    """
    cs["Cleavage"] = True
    zb["Cleavage"] = False

    merged_rows = []

    for _, row in zb.iterrows():
        matched = False
        for _, crow in cs.iterrows():
            off1, off2 = row["OffTargetSequence"], crow["OffTargetSequence"]
            tgt1, tgt2 = row["TargetSequence"], crow["TargetSequence"]

            if isinstance(off1, str) and isinstance(off2, str) and isinstance(tgt1, str) and isinstance(tgt2, str):
                if (levenshtein(off1, off2) <= max_dist and
                        levenshtein(tgt1, tgt2) <= max_dist):
                    matched = True
                    break

        if matched:
            row["Cleavage"] = True
        merged_rows.append(row)

    merged = pd.concat([cs, pd.DataFrame(merged_rows)], ignore_index=True)

    merged = merged.dropna(subset=["TargetSequence", "OffTargetSequence"])

    merged = merged.sort_values(by="Cleavage", ascending=False).reset_index(drop=True)

    return merged


def save_model_inputs(df, outdir):
    """Save master + model-specific input files."""
    os.makedirs(outdir, exist_ok=True)

    df.to_csv(os.path.join(outdir, "master_merged.csv"), index=False)

    df[["TargetSequence", "OffTargetSequence"]].to_csv(
        os.path.join(outdir, "CRISPR-NET_input.txt"),
        index=False, header=False
    )

    df[["OffTargetSequence", "TargetSequence"]].to_csv(
        os.path.join(outdir, "piCRISPR_input.csv"),
        index=False
    )

    dipoff_df = df.rename(columns={
        "TargetSequence": "sgRNA",
        "OffTargetSequence": "Target_DNA"
    })
    dipoff_df[["sgRNA", "Target_DNA"]].to_csv(
        os.path.join(outdir, "CRISPR-DIPOFF_input.csv"),
        index=False
    )


def main():
    cs, zb = load_data()
    merged = merge_datasets(cs, zb, max_dist=2)
    save_model_inputs(merged, OUTPUT_DIR)
    print(f"[info] Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
