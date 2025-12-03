from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

DEFAULT_DATA_DIR = Path("/home/kcv/Desktop/constant_power_test/post_processing/data")
DEFAULT_OUTPUT_DIR = Path("/home/kcv/Desktop/constant_power_test/post_processing/plots")
DEFAULT_OUTPUT_NAME = "chg_dchg_plots.pdf"


def color_map_for_steps(summary: pd.DataFrame) -> dict[int, tuple]:
    """Use Viridis scaled by |P|max for every step."""
    colors: dict[int, tuple] = {}

    if not summary.empty:
        powers = summary["max_power_W"]
        normalized = (
            np.full(len(powers), 0.5)
            if powers.max() == powers.min()
            else (powers - powers.min()) / (powers.max() - powers.min())
        )
        cmap = plt.colormaps["viridis"]
        for step_no, val in zip(summary.index, normalized):
            colors[step_no] = cmap(val)

    grey = (0.6, 0.6, 0.6, 0.9)
    for step_no in summary.index:
        colors.setdefault(step_no, grey)

    return colors


def is_charge_step(name: str) -> bool:
    lowered = str(name).lower()
    return "chg" in lowered


def generate_chg_dchg_plots(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    x_field: str = "Absolute time",
    y_fields: Sequence[str] = ("volt(V)", "Current(A)"),
    output_filename: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Create multi-page PDF plotting specified y-fields vs x-field for charge/discharge steps."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / output_filename

    with PdfPages(output_pdf) as pdf:
        for csv_path in sorted(data_dir.glob("*.csv")):
            if csv_path.name == "eda_summary.csv":
                continue

            df = pd.read_csv(csv_path)

            if x_field not in df.columns:
                raise KeyError(f"{x_field} not found in {csv_path.name}")
            missing_y = [col for col in y_fields if col not in df.columns]
            if missing_y:
                raise KeyError(f"{csv_path.name} missing columns: {', '.join(missing_y)}")

            if x_field == "Absolute time":
                df[x_field] = pd.to_datetime(df[x_field], errors="coerce")

            def convert_power_field(field_name: str) -> str:
                if field_name != "Power(mW)":
                    return field_name
                converted_name = "Power(W)"
                df[converted_name] = df["Power(mW)"].abs() / 1000
                return converted_name

            plot_x_field = convert_power_field(x_field)
            plot_y_fields = [convert_power_field(field) for field in y_fields]

            df = df.dropna(subset=[plot_x_field])

            for step_name, step_df in df.groupby("Step name"):
                if not is_charge_step(step_name):
                    continue

                step_df = step_df.sort_values(x_field)

                summary = (
                    step_df.groupby("Step No")
                    .agg(
                        max_power_W=("Power(mW)", lambda s: s.abs().max() / 1000),
                        duration=("duration", "max"),
                        c_rate=("c-rate", "max"),
                    )
                )

                colors = color_map_for_steps(summary)

                fig, axes = plt.subplots(
                    len(plot_y_fields),
                    1,
                    figsize=(11, 4 * len(plot_y_fields)),
                    sharex=True,
                )
                if len(plot_y_fields) == 1:
                    axes = [axes]
                fig.suptitle(f"{csv_path.stem} — {step_name}")

                for axis, field in zip(axes, plot_y_fields):
                    for step_no, group in step_df.groupby("Step No"):
                        color = colors.get(step_no, (0.6, 0.6, 0.6, 0.9))
                        c_rate = summary.loc[step_no, "c_rate"]
                        duration = summary.loc[step_no, "duration"]
                        c_rate_str = f"{c_rate:.2f} C" if pd.notna(c_rate) else None
                        duration_str = f"{duration:.1f} s" if pd.notna(duration) else None
                        legend_parts = [f"Step {step_no}"]
                        if c_rate_str:
                            legend_parts.append(c_rate_str)
                        if duration_str:
                            legend_parts.append(duration_str)
                        legend = " — ".join(legend_parts)
                        axis.scatter(
                            group[x_field],
                            group[field],
                            s=16,
                            alpha=0.85,
                            color=color,
                            label=legend if axis is axes[0] else None,
                        )

                    axis.set_ylabel(field)
                    axis.grid(True, linestyle="--", alpha=0.3)

                axes[-1].set_xlabel(plot_x_field)
                axes[0].legend(
                    fontsize="small",
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=10,
                )

                plt.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved charge/discharge plots to {output_pdf}")
    return output_pdf


if __name__ == "__main__":
    generate_chg_dchg_plots()
