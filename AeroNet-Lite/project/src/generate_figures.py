"""
generate_figures.py
Generates all static PNG figures required by the AeroNet Lite report.
Outputs to ../report/figures/

Run from project root or src/:
    python src/generate_figures.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(__file__))

from grid_model import build_grid
from layout_validator import run_validator
from fleet_selector import run_genetic_algorithm
from astar_planner import plan_delivery_route
from delivery_simulator import run_simulation
from ml_pipeline import (
    run_demand_forecasting, run_anomaly_detection,
    generate_synthetic_telemetry,
)
from sklearn.metrics import confusion_matrix

# Reproducible figures
random.seed(7)
np.random.seed(7)

FIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "report", "figures")
)
PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed")
)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

ZONE_COLORS = {
    "Residential": "#A8D5BA",
    "Commercial":  "#FFD8A8",
    "Industrial":  "#C0C0C0",
    "Hospital":    "#F4A8A8",
    "School":      "#FFE680",
    "Open Field":  "#EAF2D7",
}


# ---------------------------------------------------------------- helpers
def _draw_grid(ax, grid, title, show_flags=True):
    """Draw a 10x10 zone grid on the given axis."""
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            color = "#2C2C2C" if cell.no_fly else ZONE_COLORS.get(cell.zone, "#FFFFFF")
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor=color, edgecolor="white", linewidth=1.0))
            if show_flags:
                if cell.is_hub:
                    ax.text(c, r, "HUB", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="#0033A0")
                elif cell.is_charging:
                    ax.text(c, r, "CHG", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="#006400")
                elif cell.is_medical_pickup:
                    ax.text(c, r, "MED", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="#8B0000")
                elif cell.no_fly:
                    ax.text(c, r, "NFZ", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white")

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(9.5, -0.5)  # invert y so row 0 is on top
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.grid(False)


def _zone_legend(extra=None):
    handles = [Patch(facecolor=col, edgecolor="white", label=zone)
               for zone, col in ZONE_COLORS.items()]
    if extra:
        handles.extend(extra)
    return handles


# ---------------------------------------------------------------- figures
def fig_zone_map(grid, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_grid(ax, grid, "AeroNet Lite — Zone Map (10x10)")
    extra = [Patch(facecolor="#2C2C2C", label="No-Fly Zone")]
    ax.legend(handles=_zone_legend(extra), bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0., frameon=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_demand_heatmap(grid, out_path):
    demand = np.array([[grid[r][c].demand for c in range(10)] for r in range(10)])

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = LinearSegmentedColormap.from_list("aero_demand", ["#F4FBF7", "#2E7D32"])
    im = ax.imshow(demand, cmap=cmap, vmin=0)

    for r in range(10):
        for c in range(10):
            ax.text(c, r, f"{demand[r, c]:.1f}", ha="center", va="center",
                    fontsize=8, color="black")
            if grid[r][c].is_hub:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       fill=False, edgecolor="#0033A0", linewidth=2.0))

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Demand Heatmap (per cell, hubs outlined)", fontsize=13, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, label="Demand")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_route_map(grid, drones, out_path):
    """Overlay each drone's most recent completed/active path on the zone map."""
    fig, ax = plt.subplots(figsize=(9.5, 8))
    _draw_grid(ax, grid, "Delivery Routes & No-Fly Cells", show_flags=True)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#17becf"]
    plotted = 0
    for i, drone in enumerate(drones):
        path = drone.completed_route if drone.completed_route else drone.route
        if not path or len(path) < 2:
            continue
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        col = palette[i % len(palette)]
        ax.plot(xs, ys, "-", color=col, linewidth=2.4, alpha=0.85,
                label=f"{drone.drone_id} ({drone.drone_type})")
        ax.plot(xs[0], ys[0], "o", color=col, markersize=8, markeredgecolor="white")
        ax.plot(xs[-1], ys[-1], "s", color=col, markersize=8, markeredgecolor="white")
        plotted += 1

    extra = [
        Patch(facecolor="#2C2C2C", label="No-Fly Zone"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=8, label="Path start"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markersize=8, label="Path end"),
    ]
    handles = _zone_legend(extra)
    if plotted:
        # add drone-line entries
        for i, drone in enumerate(drones[:plotted]):
            handles.append(plt.Line2D([0], [0], color=palette[i % len(palette)],
                                       linewidth=2.4, label=f"{drone.drone_id}"))

    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0., frameon=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_ga_fitness(history, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    gens = list(range(1, len(history) + 1))
    ax.plot(gens, history, "-o", color="#0033A0", markersize=4, linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness Score")
    ax.set_title("Genetic Algorithm — Fleet Selection Convergence",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(max(history), color="#2E7D32", linestyle="--", alpha=0.6,
               label=f"Best = {max(history):.4f}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_confusion_matrix(cm, classes, model_name, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_event_log(event_log, out_path):
    """Render the event log as a clean text figure."""
    text = "\n".join(event_log)
    line_count = len(event_log)
    fig_h = max(6, line_count * 0.22)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    ax.text(0.5, 0.99, "AeroNet Lite — 20-Step Simulation Event Log",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            ha="center", va="top")
    ax.text(0.01, 0.96, text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=9, va="top", ha="left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("  GENERATING REPORT FIGURES")
    print("=" * 60)

    # 1) build grid + run all modules once so figures share one snapshot
    grid = build_grid()
    run_validator(grid)  # ensures validity (no side-effects on the grid)

    print("\n[GA] Running fleet selection...")
    fleet = run_genetic_algorithm(grid, budget=12000, population_size=30, generations=60)

    print("\n[ML] Running demand forecasting + anomaly detection...")
    demand = run_demand_forecasting()
    anomaly = run_anomaly_detection()

    print("\n[Sim] Running 20-step simulation...")
    sim = run_simulation(grid, fleet, demand_forecast=demand)
    drones = sim["drones"]
    event_log = sim["event_log"]

    # 2) export drone telemetry CSV (matches the README deliverable)
    telemetry = generate_synthetic_telemetry()
    telemetry_path = os.path.join(PROCESSED_DIR, "drone_telemetry.csv")
    telemetry.to_csv(telemetry_path, index=False)
    print(f"\n[CSV] drone_telemetry.csv written to {telemetry_path}")

    # 3) write event log as plain text too
    log_txt = os.path.join(FIG_DIR, "event_log.txt")
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(event_log))
    print(f"[Log] event_log.txt written to {log_txt}")

    # 4) figures
    print("\n[Figures] Writing PNGs to:", FIG_DIR)
    fig_zone_map(grid, os.path.join(FIG_DIR, "zone_map.png"))
    fig_demand_heatmap(grid, os.path.join(FIG_DIR, "demand_heatmap.png"))
    fig_route_map(grid, drones, os.path.join(FIG_DIR, "route_map.png"))
    fig_ga_fitness(fleet["ga_history"], os.path.join(FIG_DIR, "ga_fitness.png"))

    classes = anomaly["classes"]
    for model_name, res in anomaly["models"].items():
        out = os.path.join(FIG_DIR, f"confusion_matrix_{model_name}.png")
        fig_confusion_matrix(res["confusion_matrix"], classes, model_name, out)

    fig_event_log(event_log, os.path.join(FIG_DIR, "event_log.png"))

    # 5) summary
    print("\n" + "=" * 60)
    print("  FIGURES GENERATED")
    print("=" * 60)
    for f in sorted(os.listdir(FIG_DIR)):
        path = os.path.join(FIG_DIR, f)
        size = os.path.getsize(path) // 1024
        print(f"  {f}  ({size} KB)")
    print(f"\nDeliveries: {sim['completed']} completed, "
          f"{sim['delayed']} delayed, {sim['failed']} failed.")
    print("Done.")


if __name__ == "__main__":
    main()
