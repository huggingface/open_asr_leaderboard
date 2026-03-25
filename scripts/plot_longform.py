"""
Plot Pareto frontier for the longform ASR leaderboard.

Setup: pip install pandas matplotlib
Usage:
```
python scripts/plot_longform.py <csv_file>
python scripts/plot_longform.py <csv_file> --highlight "model_id"
```

Example:
```
python scripts/plot_longform.py scripts/data/25032026_longform.csv --highlight "nvidia/parakeet-tdt-0.6b-v3"
```
"""

import argparse

import pandas as pd
from plot_utils import plot_wer_tradeoff


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Pareto frontier for the longform ASR leaderboard."
    )
    parser.add_argument("csv_file", help="Path to the longform leaderboard CSV file.")
    parser.add_argument("--label-col", default="model_id", help="Column name for model labels (default: model_id).")
    parser.add_argument("--wer-col", default="Average", help="Column name for average WER (default: Average).")
    parser.add_argument("--rtfx-col", default="RTFx", help="Column name for RTFx values (default: RTFx).")
    parser.add_argument("--size-col", default="Model size (B)", help="Column name for model size (default: 'Model size (B)').")

    # RTFx plot limits
    parser.add_argument("--rtfx-xlim", type=float, nargs=2, default=[10, 25], metavar=("MIN", "MAX"), help="X-axis limits for the RTFx plot (default: 5 25).")
    parser.add_argument("--rtfx-ylim", type=float, nargs=2, default=[1e2, 1e4], metavar=("MIN", "MAX"), help="Y-axis limits for the RTFx plot (default: 1e1 1e4).")

    # Model size plot limits
    parser.add_argument("--size-xlim", type=float, nargs=2, default=[10, 25], metavar=("MIN", "MAX"), help="X-axis limits for the model-size plot (default: 5 25).")
    parser.add_argument("--size-ylim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Y-axis limits for the model-size plot (default: None).")
    parser.add_argument("--size-yfact", type=float, default=1e3, help="Multiplicative factor for model size values, e.g. 1e3 to convert B to M (default: 1000).")

    # Output files
    parser.add_argument("--rtfx-output", default="longform_rtfx_wer.png", help="Output filename for the RTFx plot (default: longform_rtfx_wer.png).")
    parser.add_argument("--size-output", default="longform_size_wer.png", help="Output filename for the model-size plot (default: longform_size_wer.png).")

    # Highlight specific
    parser.add_argument("--highlight", default=None, help="Model name to highlight with a star marker and red label (default: None).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.csv_file)

    # Plot 1: WER vs RTFx (longform)
    plot_wer_tradeoff(
        df=df,
        x_col=args.wer_col,
        y_col=args.rtfx_col,
        label_col=args.label_col,
        y_label="RTFx",
        x_label="Avg. WER (%)",
        output_file=args.rtfx_output,
        x_goal="min",
        y_goal="max",
        x_lim=tuple(args.rtfx_xlim),
        y_lim=tuple(args.rtfx_ylim),
        highlight_model=args.highlight,
    )

    # Plot 2: WER vs Model Size (longform)
    plot_wer_tradeoff(
        df=df,
        x_col=args.wer_col,
        y_col=args.size_col,
        label_col=args.label_col,
        y_label="Model Size (M)",
        x_label="Avg. WER (%)",
        output_file=args.size_output,
        x_goal="min",
        y_goal="min",
        x_lim=tuple(args.size_xlim),
        y_lim=tuple(args.size_ylim) if args.size_ylim else None,
        y_fact=args.size_yfact,
        highlight_model=args.highlight,
    )
