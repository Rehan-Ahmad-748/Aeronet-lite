"""
grid_model.py
Shared 10x10 grid model for AeroNet Lite.
Each cell is a Python dataclass with zone, density, flags, and demand.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import random

# Zone types
ZONE_TYPES = [
    "Residential", "Commercial", "Industrial",
    "Hospital", "School", "Open Field"
]

@dataclass
class Cell:
    row: int
    col: int
    zone: str = "Open Field"
    density: int = 0
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0

    def __repr__(self):
        flags = []
        if self.is_hub:           flags.append("HUB")
        if self.is_charging:      flags.append("CHG")
        if self.is_medical_pickup:flags.append("MED")
        if self.no_fly:           flags.append("NFZ")
        flag_str = "|".join(flags)
        return f"Cell({self.row},{self.col} {self.zone[:3]} {flag_str})"


def build_grid() -> List[List[Cell]]:
    """
    Build a fixed 10x10 grid with a realistic city layout.
    Layout is manually defined so CSP validation has meaningful rules to check.
    """
    # Zone layout (10 rows x 10 cols)
    zone_layout = [
        ["Residential","Residential","Commercial","Commercial","Industrial","Industrial","Open Field","Open Field","Residential","Residential"],
        ["Residential","School",     "Commercial","Commercial","Industrial","Open Field", "Open Field","Hospital",  "Residential","Residential"],
        ["Residential","Residential","Commercial","Open Field","Open Field","Open Field", "Open Field","Hospital",  "Residential","Residential"],
        ["Open Field", "Open Field", "Open Field","Open Field","Open Field","Open Field", "Residential","Residential","Open Field","Open Field"],
        ["Residential","Residential","Open Field","Open Field","Open Field","Open Field", "Residential","Residential","Industrial","Industrial"],
        ["Residential","Residential","Commercial","Commercial","Open Field","Open Field", "Open Field","Open Field", "Industrial","Industrial"],
        ["School",     "Residential","Residential","Residential","Residential","Commercial","Commercial","Open Field","Open Field","Open Field"],
        ["Open Field", "Residential","Residential","Residential","Residential","Commercial","Commercial","Open Field","Residential","Residential"],
        ["Open Field", "Open Field", "Open Field","Open Field","Open Field","Open Field", "Residential","Residential","Residential","Residential"],
        ["Open Field", "Open Field", "Open Field","Open Field","Open Field","Residential","Residential","Residential","Residential","Residential"],
    ]

    # Density values per zone type
    density_map = {
        "Residential": 5000, "Commercial": 3000, "Industrial": 1000,
        "Hospital": 2000, "School": 1500, "Open Field": 500
    }

    grid = []
    for r in range(10):
        row = []
        for c in range(10):
            zone = zone_layout[r][c]
            cell = Cell(
                row=r, col=c,
                zone=zone,
                density=density_map[zone] + random.randint(-200, 200)
            )
            row.append(cell)
        grid.append(row)

    # ---- Place Hubs ----
    # Positioned so every residential cell is within 3 Manhattan cells of a hub.
    hub_positions = [(0, 2), (1, 8), (3, 1), (5, 5), (7, 1), (9, 8)]
    for (r, c) in hub_positions:
        grid[r][c].is_hub = True

    # ---- Place Charging Pads (within 2 cells of each hub) ----
    charging_positions = [(0, 3), (1, 7), (3, 2), (5, 4), (7, 2), (9, 7)]
    for (r, c) in charging_positions:
        grid[r][c].is_charging = True

    # ---- Place Medical Pickups (next to hospitals) ----
    medical_positions = [(1, 8), (2, 8)]  # adjacent to hospitals at (1,7),(2,7)
    for (r, c) in medical_positions:
        grid[r][c].is_medical_pickup = True

    # ---- Set initial demand (will be overridden by ML pipeline) ----
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            if cell.zone == "Residential":
                cell.demand = round(random.uniform(3.0, 9.0), 2)
            elif cell.zone == "Commercial":
                cell.demand = round(random.uniform(2.0, 6.0), 2)
            else:
                cell.demand = round(random.uniform(0.5, 2.0), 2)

    return grid


def get_hubs(grid: List[List[Cell]]):
    return [(r, c) for r in range(10) for c in range(10) if grid[r][c].is_hub]

def get_charging_pads(grid: List[List[Cell]]):
    return [(r, c) for r in range(10) for c in range(10) if grid[r][c].is_charging]

def get_medical_pickups(grid: List[List[Cell]]):
    return [(r, c) for r in range(10) for c in range(10) if grid[r][c].is_medical_pickup]

def get_hospitals(grid: List[List[Cell]]):
    return [(r, c) for r in range(10) for c in range(10) if grid[r][c].zone == "Hospital"]

def get_residential(grid: List[List[Cell]]):
    return [(r, c) for r in range(10) for c in range(10) if grid[r][c].zone == "Residential"]

def get_industrial(grid: List[List[Cell]]):
    return [(r, c) for r in range(10) for c in range(10) if grid[r][c].zone == "Industrial"]

def get_neighbors(row: int, col: int, grid_size: int = 10):
    """Return valid 4-directional neighbors."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            neighbors.append((nr, nc))
    return neighbors

def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
