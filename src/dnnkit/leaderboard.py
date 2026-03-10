import glob
import json
import os
import pandas as pd


def build_leaderboard(outputs_dir="outputs"):
    records = []

    for metrics_file in glob.glob(os.path.join(outputs_dir, "*", "metrics.json")):
        with open(metrics_file, "r") as f:
            records.append(json.load(f))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(by="final_test_accuracy", ascending=False)
    return df


def save_leaderboard(outputs_dir="outputs"):
    df = build_leaderboard(outputs_dir=outputs_dir)
    if df.empty:
        print("No runs found.")
        return

    out_csv = os.path.join(outputs_dir, "leaderboard.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    save_leaderboard()
