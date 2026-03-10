import os
from dnnkit.leaderboard import build_leaderboard


def generate_report(outputs_dir="outputs", output_file="paper/generated_report.md"):
    df = build_leaderboard(outputs_dir=outputs_dir)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if df.empty:
        text = """# DNNKit Experimental Report

No experiments were found in the outputs directory.
Run training first, then regenerate the report.
"""
    else:
        best = df.iloc[0]
        text = f"""# DNNKit Experimental Report

## Overview
This report summarizes benchmark experiments run with DNNKit.

## Best Run
- Dataset: {best['dataset']}
- Epochs: {best['epochs']}
- Learning rate: {best['lr']}
- Batch size: {best['batch_size']}
- Final loss: {best['final_loss']:.4f}
- Final test accuracy: {best['final_test_accuracy']:.4f}
- Device: {best['device']}
- Run directory: {best['run_dir']}

## Leaderboard
{df.to_markdown(index=False)}

## Summary
The best-performing experiment achieved a final test accuracy of
{best['final_test_accuracy']:.4f} on the {best['dataset']} benchmark.
This demonstrates that DNNKit supports reproducible training,
benchmarking, and experiment comparison workflows.
"""

    with open(output_file, "w") as f:
        f.write(text)

    print(f"Saved {output_file}")
