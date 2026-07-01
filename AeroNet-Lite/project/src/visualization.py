"""
visualization.py  –  AeroNet Lite Streamlit Dashboard
Run:  streamlit run src/visualization.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from grid_model import build_grid
from layout_validator import run_validator
from fleet_selector import run_genetic_algorithm
from ml_pipeline import run_demand_forecasting, run_anomaly_detection
from delivery_simulator import run_simulation

st.set_page_config(page_title="AeroNet Lite", page_icon="🚁", layout="wide")

# ─── Pastel palette (light, clear, matches the project's reference grid) ─────
ZONE_COLORS = {
    "Residential": "#B7D5EF",   # soft sky blue
    "Commercial":  "#F5D2A8",   # peach
    "Industrial":  "#D4B595",   # warm tan
    "Hospital":    "#F5B5B5",   # blush pink
    "School":      "#A8DDB1",   # mint
    "Open Field":  "#E8E8DC",   # cream
}

ZONE_LABEL_SHORT = {
    "Residential": "Resident.",
    "Commercial":  "Comm.",
    "Industrial":  "Indust.",
    "Hospital":    "Hospital",
    "School":      "School",
    "Open Field":  "Open",
}

# Special-cell overrides — strong colors that read clearly on a white grid
HUB_FILL_COLOR      = "#1F4DC2"
CHARGING_FILL_COLOR = "#1B9F76"
MEDICAL_FILL_COLOR  = "#E0496B"
NO_FLY_FILL_COLOR   = "#1F1F1F"
NO_FLY_BORDER_COLOR = "#FF1744"

CELL_TEXT_COLOR     = "#2C3E50"          # dark slate for zone labels
CELL_BORDER_COLOR   = "rgba(0,0,0,0.18)"
SPECIAL_TEXT_COLOR  = "white"

# Light chart theme
PAPER_BG           = "#FFFFFF"
PLOT_BG            = "#FFFFFF"
AXIS_COLOR         = "rgba(0,0,0,0.55)"
GRID_LINE_COLOR    = "rgba(0,0,0,0.05)"
TITLE_COLOR        = "#1F2937"
LEGEND_BG          = "#FFFFFF"
LEGEND_BORDER      = "rgba(0,0,0,0.18)"

# Pickup / dropoff markers (route maps)
PICKUP_COLOR  = "#FF6F00"   # vivid orange
DROPOFF_COLOR = "#6A1B9A"   # deep purple
START_COLOR   = "#2E7D32"   # green for "drone starts here"
END_COLOR     = "#1565C0"   # blue for "drone ends here"

# Drone trail palette — saturated, distinguishable on a white grid
DRONE_PALETTE = ["#E91E63", "#FF6D00", "#7B1FA2", "#0277BD",
                 "#F9A825", "#388E3C", "#00695C", "#C62828"]


# ── cached pipeline (also pre-builds figures once per session) ─────────────────
@st.cache_resource(show_spinner="Running AeroNet Lite pipeline...")
def run_pipeline():
    grid    = build_grid()
    report  = run_validator(grid)
    fleet   = run_genetic_algorithm(grid, budget=12000, population_size=30, generations=60)
    demand  = run_demand_forecasting()
    anomaly = run_anomaly_detection()
    sim     = run_simulation(grid, fleet, demand_forecast=demand)

    figs = {
        "zone_map":         zone_map_fig(grid),
        "route_animation":  route_map_animation_fig(
            grid, sim["snapshots"], sim["drones"], event_log=sim["event_log"]),
        "route_static":     route_map_static_fig(grid, sim["drones"], sim["deliveries"]),
        "demand_heatmap":   demand_heatmap_fig(grid),
        "ga_fitness":       ga_fitness_fig(fleet["ga_history"]),
        "model_comparison": model_comparison_fig(demand, anomaly),
        "cm_dt": confusion_matrix_fig(
            anomaly["models"]["DecisionTree"]["confusion_matrix"],
            anomaly["classes"], "Decision Tree"),
        "cm_rf": confusion_matrix_fig(
            anomaly["models"]["RandomForest"]["confusion_matrix"],
            anomaly["classes"], "Random Forest"),
    }

    # ── Save all figures as PNG to report/figures/ ──────────────────────────
    _save_figs_to_disk(figs, sim)

    return grid, report, fleet, demand, anomaly, sim, figs


def _save_figs_to_disk(figs, sim):
    """Write every Plotly figure to report/figures/ as a static PNG.
    Requires the 'kaleido' package (pip install kaleido).
    Silently skips saving if kaleido is not installed."""
    try:
        import kaleido  # noqa: F401 – just check it's available
    except ImportError:
        import streamlit as _st
        _st.warning(
            "⚠️ Figures were NOT saved to disk. "
            "Install kaleido to enable PNG export:  pip install kaleido"
        )
        return

    fig_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "report", "figures")
    )
    os.makedirs(fig_dir, exist_ok=True)

    # Map figure keys → output filenames
    save_map = {
        "zone_map":         "zone_map.png",
        "route_static":     "route_map.png",
        "demand_heatmap":   "demand_heatmap.png",
        "ga_fitness":       "ga_fitness.png",
        "model_comparison": "anomaly_model_comparison.png",
        "cm_dt":            "confusion_matrix_decisiontree.png",
        "cm_rf":            "confusion_matrix_randomforest.png",
    }

    for key, filename in save_map.items():
        fig = figs.get(key)
        if fig is None:
            continue
        out_path = os.path.join(fig_dir, filename)
        try:
            fig.write_image(out_path, width=1400, height=900, scale=1.5)
        except Exception as exc:
            import streamlit as _st
            _st.warning(f"Could not save {filename}: {exc}")

    # Save event log as text + image
    event_log = sim.get("event_log", [])
    if event_log:
        log_txt = os.path.join(fig_dir, "event_log.txt")
        with open(log_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(event_log))

    # Also save the animated route map — Plotly HTML (animations can't be PNG)
    anim_html = os.path.join(fig_dir, "route_animation.html")
    try:
        figs["route_animation"].write_html(anim_html)
    except Exception:
        pass


# ─── helpers ─────────────────────────────────────────────────────────────────
def _cell_style(cell, is_nf):
    """Return (fill_color, label_text, text_color, is_special) for one cell.
    Priority: no_fly  >  hub  >  charging  >  medical  >  zone."""
    if is_nf:
        return NO_FLY_FILL_COLOR, "NO-FLY", SPECIAL_TEXT_COLOR, True
    if cell.is_hub:
        return HUB_FILL_COLOR, "HUB", SPECIAL_TEXT_COLOR, True
    if cell.is_charging:
        return CHARGING_FILL_COLOR, "CHG", SPECIAL_TEXT_COLOR, True
    if cell.is_medical_pickup:
        return MEDICAL_FILL_COLOR, "MED", SPECIAL_TEXT_COLOR, True
    return (ZONE_COLORS.get(cell.zone, "#FFFFFF"),
            ZONE_LABEL_SHORT.get(cell.zone, cell.zone),
            CELL_TEXT_COLOR, False)


def _hover_text(grid):
    """Build per-cell hover with zone, density, demand, flags."""
    out = []
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            flags = []
            if cell.is_hub:           flags.append("HUB")
            if cell.is_charging:      flags.append("Charging Pad")
            if cell.is_medical_pickup:flags.append("Medical Pickup")
            if cell.no_fly:           flags.append("⚠ No-Fly")
            flag_str = " • ".join(flags) if flags else "—"
            out.append(
                f"<b>({r},{c})</b><br>"
                f"Zone: <b>{cell.zone}</b><br>"
                f"Density: {cell.density}<br>"
                f"Demand: {cell.demand:.1f}<br>"
                f"Flags: {flag_str}"
            )
    return out


def _add_grid_shapes_and_labels(fig, grid, with_labels=True, label_position="center"):
    """Render the 10×10 zone grid as rectangle shapes + text annotations.

    label_position:
        "center"   — zone names centered (used for the standalone zone map
                     and demand heatmap, matches the project reference image).
        "corner"   — zone names in the top-left corner of each cell, leaving
                     the center free for drone / pickup / dropoff markers
                     (used for both route maps).
    Special cells (HUB / CHG / MED) ALWAYS render their label centered with
    a bold font, since the label is the primary info for that cell type."""
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            fill, label, text_col, is_special = _cell_style(cell, is_nf=False)

            fig.add_shape(
                type="rect",
                x0=c-0.5, x1=c+0.5, y0=r-0.5, y1=r+0.5,
                fillcolor=fill,
                line=dict(color=CELL_BORDER_COLOR, width=1),
                layer="below",
            )
            if not with_labels:
                continue

            if is_special:
                # HUB / CHG / MED — centered, bold, prominent.
                fig.add_annotation(
                    x=c, y=r, text=label,
                    showarrow=False,
                    font=dict(size=11, color=text_col, family="Arial Black"),
                )
            elif label_position == "corner":
                # Tucked into the top-left corner of the cell so the centre
                # is available for dynamic markers (drones, pickups, etc.).
                fig.add_annotation(
                    x=c-0.42, y=r-0.42, text=label,
                    showarrow=False,
                    xanchor="left", yanchor="top",
                    font=dict(size=8, color=text_col, family="Arial"),
                )
            else:
                # Centered zone label (default zone-map look).
                fig.add_annotation(
                    x=c, y=r, text=label,
                    showarrow=False,
                    font=dict(size=10, color=text_col, family="Arial"),
                )


def _no_fly_overlay_trace(no_fly_cells, marker_size=58):
    """Overlay no-fly cells with a black square + 'NO-FLY' text.
    Used per-frame in the route animation."""
    if not no_fly_cells:
        return go.Scatter(
            x=[], y=[], mode="markers",
            name="No-Fly Zone", showlegend=True,
            marker=dict(symbol="square", size=marker_size,
                        color=NO_FLY_FILL_COLOR,
                        line=dict(color=NO_FLY_BORDER_COLOR, width=2)),
            hoverinfo="skip",
        )
    return go.Scatter(
        x=[c for (_, c) in no_fly_cells],
        y=[r for (r, _) in no_fly_cells],
        mode="markers+text",
        marker=dict(symbol="square", size=marker_size,
                    color=NO_FLY_FILL_COLOR,
                    line=dict(color=NO_FLY_BORDER_COLOR, width=2)),
        text=["NO-FLY"] * len(no_fly_cells),
        textposition="middle center",
        textfont=dict(color=SPECIAL_TEXT_COLOR, size=10, family="Arial Black"),
        name="No-Fly Zone",
        hoverinfo="skip", showlegend=True,
    )


def _legend_chip_traces():
    """Invisible scatter traces that produce zone + special-cell legend chips
    on the right side of the chart (same chips as the Zone Map)."""
    chips = []
    for zone, color in ZONE_COLORS.items():
        chips.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=color,
                        line=dict(color=CELL_BORDER_COLOR, width=1)),
            name=zone, hoverinfo="skip", showlegend=True,
        ))
    for label, color in [("Hub", HUB_FILL_COLOR),
                          ("Charging Pad", CHARGING_FILL_COLOR),
                          ("Medical Pickup", MEDICAL_FILL_COLOR)]:
        chips.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=color,
                        line=dict(color="rgba(0,0,0,0.4)", width=1)),
            name=label, hoverinfo="skip", showlegend=True,
        ))
    return chips


def _grid_layout(fig, title, h=720):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(size=16, color=TITLE_COLOR, family="Arial Black"),
                   x=0.02, xanchor="left"),
        xaxis=dict(tickvals=list(range(10)), title=None,
                   zeroline=False, range=[-0.5, 9.5],
                   gridcolor=GRID_LINE_COLOR, color=AXIS_COLOR,
                   tickfont=dict(size=11, color=AXIS_COLOR),
                   showline=False),
        yaxis=dict(tickvals=list(range(10)), title=None,
                   zeroline=False, scaleanchor="x", scaleratio=1,
                   autorange="reversed", range=[-0.5, 9.5],
                   gridcolor=GRID_LINE_COLOR, color=AXIS_COLOR,
                   tickfont=dict(size=11, color=AXIS_COLOR),
                   showline=False),
        height=h, margin=dict(l=40, r=210, t=70, b=40),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER,
                    borderwidth=1, font=dict(color=TITLE_COLOR, size=11),
                    x=1.01, xanchor="left", y=1.0, yanchor="top"),
    )


# ─── chart builders ──────────────────────────────────────────────────────────
def zone_map_fig(grid):
    fig = go.Figure()
    _add_grid_shapes_and_labels(fig, grid, with_labels=True)

    # Invisible hover layer over every cell.
    xs = [c for r in range(10) for c in range(10)]
    ys = [r for r in range(10) for c in range(10)]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=42, opacity=0),
        hovertext=_hover_text(grid), hoverinfo="text",
        showlegend=False,
    ))

    # Legend chips for the 6 zone colours + special cells.
    for zone, color in ZONE_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=color,
                        line=dict(color=CELL_BORDER_COLOR, width=1)),
            name=zone, hoverinfo="skip", showlegend=True,
        ))
    for label, color in [("Hub", HUB_FILL_COLOR),
                          ("Charging", CHARGING_FILL_COLOR),
                          ("Medical", MEDICAL_FILL_COLOR),
                          ("No-Fly", NO_FLY_FILL_COLOR)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=color,
                        line=dict(color="rgba(0,0,0,0.4)", width=1)),
            name=label, hoverinfo="skip", showlegend=True,
        ))

    _grid_layout(fig, "City Zone Map — 10 × 10 Region", h=720)
    return fig




# ── Lane-offset helper ────────────────────────────────────────────────────────
# Simple diagonal shift: each drone gets a fixed (dx, dy) added to EVERY
# point in its path.  Lines stay perfectly straight - no normals, no curves.
#
# Offsets arranged in a small diagonal grid centred on (0,0):
#   Drone 0: ( 0.00,  0.00)   Drone 1: (+0.13, +0.13)
#   Drone 2: (-0.13, -0.13)   Drone 3: (+0.13, -0.13)
#   Drone 4: (-0.13, +0.13)   Drone 5: (+0.22,  0.00) ... etc.

_SHIFT_PAIRS = [
    ( 0.00,  0.00),
    ( 0.13,  0.13),
    (-0.13, -0.13),
    ( 0.13, -0.13),
    (-0.13,  0.13),
    ( 0.22,  0.00),
    ( 0.00,  0.22),
    (-0.22,  0.00),
    ( 0.00, -0.22),
    ( 0.22,  0.22),
    (-0.22,  0.22),
]


def _lane_offsets(n: int):
    """Return list of (dx, dy) tuples, one per drone."""
    return [_SHIFT_PAIRS[i % len(_SHIFT_PAIRS)] for i in range(n)]


def _offset_path(path, shift):
    """
    Add a fixed (dx, dy) to every cell in path.
    Lines remain perfectly straight - the whole path shifts diagonally.
    Returns (xs, ys) lists ready for Plotly.
    """
    dx, dy = shift if isinstance(shift, tuple) else (shift, shift)
    xs = [p[1] + dx for p in path]
    ys = [p[0] + dy for p in path]
    return xs, ys


def _offset_point(cell, shift, _unused=None):
    """Return (x, y) for a single cell with shift applied."""
    dx, dy = shift if isinstance(shift, tuple) else (shift, shift)
    return cell[1] + dx, cell[0] + dy


# ── Fixed static route map ────────────────────────────────────────────────────

def route_map_static_fig(grid, drones, deliveries=None):
    """
    Static overview of all drone routes.

    FIXES:
      1. Each drone gets a unique lane offset so overlapping paths are
         visually separated instead of drawn on top of each other.
      2. Every drone has its OWN coloured pickup (▲) and dropoff (▼) markers
         labelled with the drone ID + delivery ID, so you can instantly see
         which drone serves which delivery.
      3. Hub-start (●) and hub-return (■) markers are drawn per drone in
         that drone's colour with its ID embedded in the label.
    """
    fig = go.Figure()
    _add_grid_shapes_and_labels(fig, grid, with_labels=True, label_position="corner")

    deliveries = deliveries or []
    delivery_by_drone = {}
    for d in deliveries:
        if d.assigned_drone:
            delivery_by_drone.setdefault(d.assigned_drone, []).append(d)

    active_drones = [d for d in drones
                     if (d.completed_route or d.route) and
                     len(d.completed_route or d.route) >= 2]

    offsets = _lane_offsets(len(active_drones))

    for i, drone in enumerate(active_drones):
        route  = drone.completed_route or drone.route
        color  = DRONE_PALETTE[i % len(DRONE_PALETTE)]
        offset = offsets[i]

        xs_off, ys_off = _offset_path(route, offset)

        # White halo for legibility
        fig.add_trace(go.Scatter(
            x=xs_off, y=ys_off, mode="lines",
            line=dict(color="white", width=9),
            opacity=0.55, hoverinfo="skip", showlegend=False,
        ))
        # Drone path line
        fig.add_trace(go.Scatter(
            x=xs_off, y=ys_off, mode="lines+markers",
            line=dict(color=color, width=5),
            marker=dict(symbol="circle", size=5, color=color,
                        line=dict(color="white", width=1)),
            name=f"{drone.drone_id} ({drone.drone_type})",
            hovertext=[
                f"<b>{drone.drone_id}</b> ({drone.drone_type})<br>"
                f"Step {k+1}/{len(route)}<br>Cell ({route[k][0]},{route[k][1]})"
                for k in range(len(route))
            ],
            hoverinfo="text",
        ))

        # ── Hub START marker (green circle) ─────────────────────────────────
        sx, sy = xs_off[0], ys_off[0]
        fig.add_trace(go.Scatter(
            x=[sx], y=[sy], mode="markers+text",
            marker=dict(symbol="circle", size=26, color=color,
                        line=dict(color="white", width=3)),
            text=[f"▶ {drone.drone_id}"],
            textposition="middle center",
            textfont=dict(color="white", size=8, family="Arial Black"),
            hovertext=(f"<b>{drone.drone_id} HUB START</b><br>"
                       f"Cell: ({route[0][0]},{route[0][1]})<br>"
                       f"Type: {drone.drone_type}"),
            hoverinfo="text", showlegend=False,
        ))

        # ── Hub RETURN marker (square in same drone colour) ──────────────────
        ex, ey = xs_off[-1], ys_off[-1]
        fig.add_trace(go.Scatter(
            x=[ex], y=[ey], mode="markers+text",
            marker=dict(symbol="square", size=22, color=color,
                        line=dict(color="white", width=2), opacity=0.85),
            text=[f"■ {drone.drone_id}"],
            textposition="middle center",
            textfont=dict(color="white", size=8, family="Arial Black"),
            hovertext=(f"<b>{drone.drone_id} RETURN</b><br>"
                       f"Cell: ({route[-1][0]},{route[-1][1]})"),
            hoverinfo="text", showlegend=False,
        ))

        # ── Per-drone PICKUP markers (▲ diamond, drone colour) ───────────────
        for dlv in delivery_by_drone.get(drone.drone_id, []):
            # Slight offset so two drones at same cell don't overlap
            px, py = _offset_point(dlv.pickup, offset)
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode="markers+text",
                marker=dict(symbol="diamond", size=24,
                            color=color,
                            line=dict(color="white", width=2)),
                text=[f"▲P{dlv.delivery_id}"],
                textposition="top center",
                textfont=dict(color=color, size=9, family="Arial Black"),
                hovertext=(f"<b>PICKUP</b><br>"
                           f"Drone: {drone.drone_id}<br>"
                           f"Delivery: {dlv.delivery_id}<br>"
                           f"Cell: {dlv.pickup}<br>"
                           f"Weight: {dlv.weight_kg} kg"),
                hoverinfo="text",
                name=f"{drone.drone_id} Pickup D{dlv.delivery_id}",
                showlegend=True,
            ))

            # ── Per-drone DROPOFF markers (▼ square, drone colour lighter) ───
            dx2, dy2 = _offset_point(dlv.dropoff, offset)
            fig.add_trace(go.Scatter(
                x=[dx2], y=[dy2], mode="markers+text",
                marker=dict(symbol="triangle-down", size=24,
                            color=color, opacity=0.75,
                            line=dict(color="white", width=2)),
                text=[f"▼D{dlv.delivery_id}"],
                textposition="bottom center",
                textfont=dict(color=color, size=9, family="Arial Black"),
                hovertext=(f"<b>DROPOFF</b><br>"
                           f"Drone: {drone.drone_id}<br>"
                           f"Delivery: {dlv.delivery_id}<br>"
                           f"Cell: {dlv.dropoff}<br>"
                           f"Status: {dlv.status}"),
                hoverinfo="text",
                name=f"{drone.drone_id} Dropoff D{dlv.delivery_id}",
                showlegend=True,
            ))

    # Legend chips for zones + special cells
    for chip in _legend_chip_traces():
        fig.add_trace(chip)

    # Manual legend entries explaining the marker shapes
    for label, sym, col in [
        ("▶ Hub Start  (circle)", "circle",        "#2E7D32"),
        ("■ Hub Return (square)", "square",         "#1565C0"),
        ("▲ Pickup     (diamond)","diamond",        PICKUP_COLOR),
        ("▼ Dropoff    (tri-dn)", "triangle-down",  DROPOFF_COLOR),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=sym, size=12, color=col,
                        line=dict(color="white", width=1)),
            name=label, hoverinfo="skip", showlegend=True,
        ))

    _grid_layout(fig, "Route Map — All Drone Paths (lane-separated)", h=800)
    return fig


# ── Fixed animated route map ──────────────────────────────────────────────────

def route_map_animation_fig(grid, snapshots, drones, event_log=None):
    """
    Frame-based animated route map.

    FIXES (same lane-offset logic applied per frame):
      1. Each drone trail is offset so overlapping paths fan out into
         separate visual lanes — you can follow each drone independently.
      2. Active pickup / dropoff markers are drawn per drone in that
         drone's own colour with drone-ID + delivery-ID labels.
      3. Hub-start markers are shown in the drone's colour.
    """
    drone_ids   = [d.drone_id for d in drones]
    drone_types = {d.drone_id: d.drone_type for d in drones}
    drone_color = {d.drone_id: DRONE_PALETTE[i % len(DRONE_PALETTE)]
                   for i, d in enumerate(drones)}
    N      = len(drone_ids)
    off_map = {did: offsets_val
               for did, offsets_val in zip(drone_ids, _lane_offsets(N))}

    # ── per-frame trace builders ───────────────────────────────────────────

    def _trail_traces(d_snap, color, drone_id):
        """Returns (halo_trace, trail_trace) with lane offset applied."""
        trail  = d_snap["trail"] if d_snap else []
        offset = off_map[drone_id]
        if trail and len(trail) >= 2:
            xs, ys = _offset_path(trail, offset)
        else:
            xs = [p[1] for p in trail]
            ys = [p[0] for p in trail]

        halo = go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="white", width=9),
            opacity=0.5, hoverinfo="skip", showlegend=False,
        )
        trail_t = go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color, width=5),
            marker=dict(symbol="circle", size=4, color=color,
                        line=dict(color="white", width=1)),
            opacity=0.95, hoverinfo="skip",
            showlegend=False, legendgroup=drone_id,
        )
        return halo, trail_t

    def _remaining_trace(d_snap, color, drone_id):
        rem    = d_snap["remaining"] if d_snap else []
        offset = off_map[drone_id]
        if d_snap and rem:
            full = [d_snap["pos"]] + list(rem)
        else:
            full = []
        if full and len(full) >= 2:
            xs, ys = _offset_path(full, offset)
        else:
            xs = [p[1] for p in full]
            ys = [p[0] for p in full]
        return go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=2.5, dash="dot"),
            opacity=0.55, hoverinfo="skip",
            showlegend=False, legendgroup=drone_id,
        )

    def _marker_trace(d_snap, color, drone_id, drone_type):
        offset = off_map[drone_id]
        if d_snap is None:
            return go.Scatter(
                x=[], y=[], mode="markers",
                name=f"{drone_id} ({drone_type})",
                legendgroup=drone_id, showlegend=True,
                marker=dict(symbol="circle", size=24, color=color),
                hoverinfo="skip",
            )
        pos    = d_snap["pos"]
        ox, oy = _offset_point(pos, offset)
        anom   = f"⚠ {d_snap['anomaly']}" if d_snap["anomaly"] else "No anomaly"
        return go.Scatter(
            x=[ox], y=[oy], mode="markers+text",
            marker=dict(symbol="circle", size=30, color=color,
                        line=dict(color="white", width=3)),
            text=[drone_id], textposition="middle center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            name=f"{drone_id} ({drone_type})",
            hovertext=(
                f"<b>{drone_id}</b> ({drone_type})<br>"
                f"Status: {d_snap['status']}<br>"
                f"Pos: {pos}<br>"
                f"Battery: {d_snap['battery']}%<br>{anom}"
            ),
            hoverinfo="text",
            legendgroup=drone_id, showlegend=True,
        )

    def _pickup_dropoff_traces(snap):
        """
        Per-drone coloured pickup (diamond) and dropoff (triangle-down).
        Each marker labelled 'DxPy' (Drone x, delivery y) so they are
        always distinguishable even when stacked at the same cell.
        """
        dlv_by_id   = {d["id"]: d for d in snap["deliveries"]}
        drone_map   = {d["id"]: d for d in snap["drones"]}
        traces      = []

        for d_snap in snap["drones"]:
            did    = d_snap["id"]
            dlv_id = d_snap.get("delivery")
            if not dlv_id:
                continue
            dlv    = dlv_by_id.get(dlv_id)
            if not dlv:
                continue
            color  = drone_color[did]
            offset = off_map[did]

            px, py = _offset_point(dlv["pickup"], offset)
            dx2, dy2 = _offset_point(dlv["dropoff"], offset)

            traces.append(go.Scatter(
                x=[px], y=[py], mode="markers+text",
                marker=dict(symbol="diamond", size=22, color=color,
                            line=dict(color="white", width=2)),
                text=[f"▲{did[1:]}P{dlv_id}"],
                textposition="top center",
                textfont=dict(color=color, size=8, family="Arial Black"),
                hovertext=f"📦 PICKUP<br>Drone: {did}<br>Del: {dlv_id}<br>Cell: {dlv['pickup']}",
                hoverinfo="text",
                name=f"{did} Pickup", showlegend=False,
            ))
            traces.append(go.Scatter(
                x=[dx2], y=[dy2], mode="markers+text",
                marker=dict(symbol="triangle-down", size=22, color=color,
                            opacity=0.8, line=dict(color="white", width=2)),
                text=[f"▼{did[1:]}D{dlv_id}"],
                textposition="bottom center",
                textfont=dict(color=color, size=8, family="Arial Black"),
                hovertext=f"🏠 DROPOFF<br>Drone: {did}<br>Del: {dlv_id}<br>Cell: {dlv['dropoff']}",
                hoverinfo="text",
                name=f"{did} Dropoff", showlegend=False,
            ))
        return traces

    # ── events annotation ──────────────────────────────────────────────────
    events_by_step = {}
    if event_log:
        for line in event_log:
            if line.startswith("Step "):
                try:
                    n = int(line.split(":", 1)[0].split()[1])
                except (ValueError, IndexError):
                    continue
                events_by_step.setdefault(n, []).append(
                    line.split(":", 1)[1].strip())

    def _events_annotation(snap):
        events = events_by_step.get(snap["step"], [])
        if not events:
            text = "<i>(no new events at this step)</i>"
        else:
            shown = events[:6]
            lines = [e if len(e) <= 95 else e[:92] + "…" for e in shown]
            if len(events) > 6:
                lines.append(f"<i>… and {len(events)-6} more</i>")
            text = "<br>".join(lines)
        return dict(
            xref="paper", yref="paper",
            x=0.0, y=-0.34, xanchor="left", yanchor="top",
            showarrow=False, align="left",
            bgcolor="#F5F8FB",
            bordercolor="rgba(0,0,0,0.18)", borderwidth=1, borderpad=10,
            font=dict(family="ui-monospace, monospace", size=11, color="#1F2937"),
            text=f"<b>📜 Step {snap['step']:02d} events</b><br>{text}",
        )

    def _frame_title(snap):
        s = snap["summary"]
        return (f"Step {snap['step']:02d}  ·  "
                f"<span style='color:#2E7D32'>✓ {s['completed']} done</span>  "
                f"<span style='color:#FF8A00'>⏳ {s['active']} active</span>  "
                f"<span style='color:#D32F2F'>✕ {s['delayed']+s['failed']}</span>  ·  "
                f"{len(snap['no_fly_cells'])} no-fly cell(s)")

    static_chips = _legend_chip_traces()

    def _frame_traces(snap):
        snap_drones = {d["id"]: d for d in snap["drones"]}
        traces = []
        # 0: no-fly overlay
        traces.append(_no_fly_overlay_trace(snap["no_fly_cells"]))
        # per-drone pickup / dropoff (variable count — one pair per active delivery)
        traces.extend(_pickup_dropoff_traces(snap))
        # per drone: halo, trail, remaining, position marker
        for did in drone_ids:
            h, t = _trail_traces(snap_drones.get(did), drone_color[did], did)
            traces.append(h)
            traces.append(t)
        for did in drone_ids:
            traces.append(_remaining_trace(snap_drones.get(did),
                                           drone_color[did], did))
        for did in drone_ids:
            traces.append(_marker_trace(snap_drones.get(did),
                                        drone_color[did], did, drone_types[did]))
        traces.extend(static_chips)
        return traces

    # Deduplicate snapshots (keep last per step)
    by_step = {}
    for s in snapshots:
        by_step[s["step"]] = s
    snapshots = [by_step[k] for k in sorted(by_step)]

    initial = _frame_traces(snapshots[0])
    fig = go.Figure(data=initial, frames=[])
    _add_grid_shapes_and_labels(fig, grid, with_labels=True, label_position="corner")
    cell_annotations = list(fig.layout.annotations)

    fig.frames = [
        go.Frame(
            data=_frame_traces(snap),
            name=str(snap["step"]),
            layout=go.Layout(
                title=dict(text=_frame_title(snap),
                           font=dict(size=15, color=TITLE_COLOR,
                                     family="Arial Black")),
                annotations=list(cell_annotations) + [_events_annotation(snap)],
            ),
        )
        for snap in snapshots
    ]

    fig.update_layout(
        title=dict(text=_frame_title(snapshots[0]),
                   font=dict(size=15, color=TITLE_COLOR, family="Arial Black"),
                   x=0.02, xanchor="left"),
        annotations=list(cell_annotations) + [_events_annotation(snapshots[0])],
        sliders=[dict(
            active=0,
            x=0.0, y=-0.08, len=1.0, xanchor="left", yanchor="top",
            pad=dict(t=14, b=14), bgcolor="#F5F8FB",
            bordercolor="rgba(0,0,0,0.18)",
            font=dict(color=TITLE_COLOR, size=11),
            currentvalue=dict(prefix="Step ", visible=True,
                              font=dict(color=TITLE_COLOR, size=13)),
            steps=[dict(
                args=[[str(snap["step"])], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                }],
                label=str(snap["step"]),
                method="animate",
            ) for snap in snapshots],
        )],
    )
    fig.update_layout(
        xaxis=dict(tickvals=list(range(10)), title=None,
                   zeroline=False, range=[-0.5, 9.5],
                   gridcolor=GRID_LINE_COLOR, color=AXIS_COLOR,
                   tickfont=dict(size=11, color=AXIS_COLOR), showline=False),
        yaxis=dict(tickvals=list(range(10)), title=None,
                   zeroline=False, scaleanchor="x", scaleratio=1,
                   autorange="reversed", range=[-0.5, 9.5],
                   gridcolor=GRID_LINE_COLOR, color=AXIS_COLOR,
                   tickfont=dict(size=11, color=AXIS_COLOR), showline=False),
        height=1000, margin=dict(l=40, r=220, t=80, b=420),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER,
                    borderwidth=1, font=dict(color=TITLE_COLOR, size=11),
                    x=1.01, xanchor="left", y=1.0, yanchor="top"),
    )
    return fig


def demand_heatmap_fig(grid):
    """Demand heatmap rendered as a styled grid (matches the zone-map look),
    cells coloured by demand intensity along a 3-stop ramp."""
    demands = [grid[r][c].demand for r in range(10) for c in range(10)]
    vmin, vmax = min(demands), max(demands)
    rng = max(vmax - vmin, 1e-6)

    def _hex_to_rgb(h):
        return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)

    def _ramp(t):
        # Light → mid orange → deep red (works on a white grid)
        stops = [(0.0, "#FFF3DC"), (0.5, "#FF9A4D"), (1.0, "#C62828")]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= t <= t1:
                local = (t - t0) / (t1 - t0)
                r0, g0, b0 = _hex_to_rgb(c0)
                r1, g1, b1 = _hex_to_rgb(c1)
                return f"#{int(r0 + (r1 - r0) * local):02X}" \
                       f"{int(g0 + (g1 - g0) * local):02X}" \
                       f"{int(b0 + (b1 - b0) * local):02X}"
        return stops[-1][1]

    fig = go.Figure()
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            t = (cell.demand - vmin) / rng
            fill = _ramp(t)
            fig.add_shape(
                type="rect",
                x0=c-0.5, x1=c+0.5, y0=r-0.5, y1=r+0.5,
                fillcolor=fill,
                line=dict(color=CELL_BORDER_COLOR, width=1),
                layer="below",
            )
            text_color = "white" if t > 0.55 else "#1F2937"
            fig.add_annotation(
                x=c, y=r, text=f"{cell.demand:.1f}",
                showarrow=False,
                font=dict(size=11, color=text_color, family="Arial"),
            )

    # Hover layer.
    xs = [c for r in range(10) for c in range(10)]
    ys = [r for r in range(10) for c in range(10)]
    ht = []
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            flags = []
            if cell.is_hub:           flags.append("HUB")
            if cell.is_charging:      flags.append("Charging")
            if cell.is_medical_pickup:flags.append("Medical")
            if cell.no_fly:           flags.append("⚠ No-Fly")
            ht.append(
                f"<b>({r},{c})</b><br>"
                f"Zone: <b>{cell.zone}</b><br>"
                f"Demand: <b>{cell.demand:.1f}</b><br>"
                f"Density: {cell.density}<br>"
                f"Flags: {' • '.join(flags) if flags else '—'}"
            )
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=42, opacity=0),
        hovertext=ht, hoverinfo="text", showlegend=False,
    ))

    # Hub overlay (so users can correlate hot demand with hub locations).
    hubs = [(r, c) for r in range(10) for c in range(10) if grid[r][c].is_hub]
    if hubs:
        fig.add_trace(go.Scatter(
            x=[c for (_, c) in hubs], y=[r for (r, _) in hubs],
            mode="markers+text",
            marker=dict(symbol="square", size=42, color=HUB_FILL_COLOR,
                        line=dict(color="white", width=2)),
            text=["HUB"] * len(hubs), textposition="middle center",
            textfont=dict(color=SPECIAL_TEXT_COLOR, size=11, family="Arial Black"),
            name="Hub", hoverinfo="skip", showlegend=True,
        ))

    # Continuous color-bar legend.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            size=0.0001,
            color=[vmin, vmax],
            colorscale=[[0.0, "#FFF3DC"], [0.5, "#FF9A4D"], [1.0, "#C62828"]],
            cmin=vmin, cmax=vmax, showscale=True,
            colorbar=dict(
                title=dict(text="Demand", font=dict(color=TITLE_COLOR)),
                tickfont=dict(color=AXIS_COLOR),
                outlinewidth=0, thickness=12, len=0.6,
                x=1.18, xanchor="left",
            ),
        ),
        hoverinfo="skip", showlegend=False,
    ))

    _grid_layout(fig, "Delivery Demand Heatmap — per-cell intensity", h=720)
    return fig


def ga_fitness_fig(history):
    fig = go.Figure(go.Scatter(
        y=history, x=list(range(1, len(history)+1)), mode="lines+markers",
        line=dict(color="#1F4DC2", width=2.5),
        marker=dict(size=5, color="#1F4DC2"),
        fill="tozeroy", fillcolor="rgba(31,77,194,0.15)"))
    fig.update_layout(
        title=dict(text="GA — Fitness over Generations",
                   font=dict(color=TITLE_COLOR, family="Arial Black")),
        xaxis=dict(title="Generation", color=AXIS_COLOR,
                   gridcolor=GRID_LINE_COLOR),
        yaxis=dict(title="Best Fitness Score", color=AXIS_COLOR,
                   gridcolor=GRID_LINE_COLOR),
        height=380, plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TITLE_COLOR),
    )
    return fig


def confusion_matrix_fig(cm, class_names, model_name):
    cm_arr = np.array(cm)
    text = [[str(cm_arr[i][j]) for j in range(len(class_names))]
            for i in range(len(class_names))]
    fig = go.Figure(go.Heatmap(
        z=cm_arr, x=class_names, y=class_names,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=True))
    fig.update_layout(
        title=dict(text=f"{model_name} — Confusion Matrix",
                   font=dict(color=TITLE_COLOR, family="Arial Black")),
        xaxis_title="Predicted", yaxis_title="Actual",
        height=420,
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TITLE_COLOR),
    )
    return fig


def model_comparison_fig(demand_result, anomaly_result):
    dm = list(demand_result["models"].keys())
    maes = [demand_result["models"][m]["MAE"] for m in dm]
    am = list(anomaly_result["models"].keys())
    accs = [anomaly_result["models"][m]["accuracy"] for m in am]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Demand — MAE (lower is better)",
                        "Anomaly — Accuracy (higher is better)"])
    fig.add_trace(go.Bar(x=dm, y=maes,
                         marker_color=["#0277BD", "#388E3C"]), row=1, col=1)
    fig.add_trace(go.Bar(x=am, y=accs,
                         marker_color=["#FF8A00", "#7B1FA2"]), row=1, col=2)
    fig.update_layout(
        height=420, showlegend=False,
        title=dict(text="ML Model Comparison",
                   font=dict(color=TITLE_COLOR, family="Arial Black")),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TITLE_COLOR),
    )
    return fig


# ─── dashboard ───────────────────────────────────────────────────────────────
def main():
    st.title("AeroNet Lite — Autonomous Drone Delivery Simulation")
    st.markdown("**BS Data Science | AI Semester Project SP2026 | FAST-NUCES**")
    st.markdown("---")

    grid, report, fleet, demand, anomaly, sim, figs = run_pipeline()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Layout Valid",     "YES" if report["valid"] else "NO",
              f"{len(report['passed'])}/4 rules passed")
    k2.metric("Fleet Size",       f"{fleet['total_drones']} drones",
              f"{fleet['light_drones']}L + {fleet['heavy_drones']}H")
    k3.metric("Budget Used",      f"${fleet['total_cost']}",
              f"${fleet['budget_remaining']} remaining")
    k4.metric("Deliveries Done",  f"{sim['completed']}/{sim['total']}",
              f"{sim['delayed']} delayed")
    k5.metric("Anomaly Accuracy", f"{anomaly['models']['RandomForest']['accuracy']:.1%}",
              "Random Forest")
    st.markdown("---")

    tabs = st.tabs(["Zone Map", "Route Map", "Demand Heatmap",
                    "Fleet (GA)", "ML Results", "Anomaly View",
                    "Simulation Log"])

    # ── Tab 1: Zone Map ────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("City Grid · Zone Map")
        st.caption("Each cell shows its zone label. Special cells take a strong "
                   "fill colour: HUB (blue), CHG (green), MED (pink), NO-FLY (black).")
        st.plotly_chart(figs["zone_map"], width="stretch")

        st.subheader("CSP Layout Validation")
        if report["valid"]:
            st.success("All 4 layout rules passed.")
        else:
            st.warning(f"{len(report['failed'])} rule(s) failed.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Passed Rules**")
            for r in report["passed"]:
                st.success(r)
        with c2:
            st.markdown("**Failed Rules**")
            for r in report["failed"]:
                st.error(r)
                for err in report["rule_results"][r]:
                    st.markdown(f"- **Error:** {err['detail']}")
                    st.markdown(f"  **Fix:** {err['fix']}")

    # ── Tab 2: Route Map (animated) ────────────────────────────────────────
    with tabs[1]:
        st.subheader("Drone Route Map — Step-by-Step Simulation")
        st.caption("Drag the step slider beneath the chart. "
                   "**Green circles** mark drones, **orange diamonds** mark "
                   "active pickups, **purple squares** mark dropoffs. "
                   "Trail = solid, remaining route = dotted. "
                   "All interaction is client-side — no page reloads.")

        if sim.get("snapshots"):
            st.plotly_chart(figs["route_animation"], width="stretch",
                            key="route_anim")
        else:
            st.info("No snapshot data — run the pipeline first.")

        with st.expander("Show all paths (static overview)", expanded=False):
            st.plotly_chart(figs["route_static"], width="stretch",
                            key="route_static")

        st.subheader("Drone Status")
        all_drones = sim["drones"]
        st.dataframe(pd.DataFrame([{
            "Drone ID":    d.drone_id,
            "Type":        d.drone_type,
            "Status":      d.status,
            "Position":    str(d.position),
            "Battery (%)": f"{d.battery:.1f}",
            "Route Len":   len(d.completed_route or d.route),
            "Anomaly":     d.anomaly or "None",
        } for d in all_drones]), width="stretch")

    # ── Tab 3: Demand Heatmap ──────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Delivery Demand Heatmap")
        st.plotly_chart(figs["demand_heatmap"], width="stretch")
        st.info(f"Predicted avg demand: **{demand['avg_demand']:.1f} units**  "
                f"| RF MAE={demand['models']['RandomForest']['MAE']}  "
                f"RMSE={demand['models']['RandomForest']['RMSE']}")

    # ── Tab 4: Fleet (GA) ──────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Fleet Selection — Genetic Algorithm")
        fa, fb = st.columns(2)
        with fa:
            st.markdown("### Selected Fleet")
            st.markdown(f"- **Light Drones:** {fleet['light_drones']} × $1000 = **${fleet['light_drones']*1000}**")
            st.markdown(f"- **Heavy Drones:** {fleet['heavy_drones']} × $1800 = **${fleet['heavy_drones']*1800}**")
            st.markdown(f"- **Total Cost:** ${fleet['total_cost']} / $12,000")
            st.markdown(f"- **Remaining:** ${fleet['budget_remaining']}")
            st.markdown(f"- **Fitness Score:** {fleet['fitness_score']}")
        with fb:
            if fleet.get("ga_history"):
                st.plotly_chart(figs["ga_fitness"], width="stretch")

    # ── Tab 5: ML Results ──────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("ML Pipeline Results")
        st.plotly_chart(figs["model_comparison"], width="stretch")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Demand Forecasting")
            st.dataframe(pd.DataFrame([
                {"Model": m, "MAE": v["MAE"], "RMSE": v["RMSE"]}
                for m, v in demand["models"].items()]), width="stretch")
        with col2:
            st.markdown("### Anomaly Detection")
            st.dataframe(pd.DataFrame([
                {"Model": m, "Accuracy": f"{v['accuracy']:.4f}"}
                for m, v in anomaly["models"].items()]), width="stretch")
        st.markdown("### Confusion Matrices")
        cm1, cm2 = st.columns(2)
        with cm1:
            st.plotly_chart(figs["cm_dt"], width="stretch")
        with cm2:
            st.plotly_chart(figs["cm_rf"], width="stretch")

    # ── Tab 6: Anomaly View ────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Anomaly Detection — Live View")
        st.caption("This tab shows the four anomaly classes the classifier "
                   "watches for, the anomalies actually detected during this "
                   "simulation run, and the actions taken in response.")

        st.markdown("### Anomaly Classes")
        st.dataframe(pd.DataFrame([
            {"Class": "Normal",
             "Description": "Battery drops gradually, low route deviation",
             "Signal Features": "battery_drop low, route_deviation low"},
            {"Class": "Battery Anomaly",
             "Description": "Battery drops faster than expected",
             "Signal Features": "battery_drop high"},
            {"Class": "Route Anomaly",
             "Description": "Drone deviates significantly from planned path",
             "Signal Features": "route_deviation high"},
            {"Class": "Sensor Spike",
             "Description": "Altitude or speed jumps suddenly",
             "Signal Features": "altitude_change / speed_change high"},
        ]), width="stretch", hide_index=True)

        # Anomalies actually detected in this simulation run
        st.markdown("### Anomalies Detected in This Run")
        anomaly_drones = [d for d in sim["drones"] if d.anomaly]
        if anomaly_drones:
            st.warning(f"{len(anomaly_drones)} drone(s) flagged with anomalies.")
            st.dataframe(pd.DataFrame([{
                "Drone":     d.drone_id,
                "Type":      d.drone_type,
                "Anomaly":   d.anomaly,
                "Battery":   f"{d.battery:.1f}%",
                "Status":    d.status,
                "Position":  str(d.position),
                "Action":    "ReturnHub" if d.battery < 20 else "Continue with monitoring",
            } for d in anomaly_drones]), width="stretch", hide_index=True)
        else:
            st.info("No anomalies were detected in this simulation run.")

        # Anomaly-related events from the log
        st.markdown("### Anomaly-Related Event Log")
        kw = ("anomaly", "battery", "return", "emergency")
        anomaly_events = [e for e in sim["event_log"]
                          if any(k in e.lower() for k in kw)]
        if anomaly_events:
            for e in anomaly_events:
                st.code(e, language="text")
        else:
            st.info("No anomaly-related events fired during the simulation.")

        # Classifier performance summary
        st.markdown("### Classifier Performance (on 950-row synthetic telemetry)")
        ma, mb = st.columns(2)
        ma.metric("Decision Tree Accuracy",
                  f"{anomaly['models']['DecisionTree']['accuracy']:.2%}")
        mb.metric("Random Forest Accuracy",
                  f"{anomaly['models']['RandomForest']['accuracy']:.2%}")
        st.plotly_chart(figs["cm_rf"], width="stretch", key="anomaly_cm")

    # ── Tab 7: Simulation Log ──────────────────────────────────────────────
    with tabs[6]:
        st.subheader("20-Step Simulation Event Log")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Completed",   sim["completed"])
        s2.metric("Delayed",     sim["delayed"])
        s3.metric("Failed",      sim["failed"])
        s4.metric("In-Progress", sim["pending"])
        st.dataframe(pd.DataFrame({"Event": sim["event_log"]}),
                     width="stretch", height=480)
        st.subheader("Delivery Status")
        st.dataframe(pd.DataFrame([{
            "ID":       d.delivery_id,
            "Pickup":   str(d.pickup),
            "Dropoff":  str(d.dropoff),
            "Weight":   f"{d.weight_kg} kg",
            "Status":   d.status,
            "Assigned": d.assigned_drone or "-",
        } for d in sim["deliveries"]]), width="stretch")


if __name__ == "__main__":
    main()