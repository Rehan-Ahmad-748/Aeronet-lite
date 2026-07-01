"""
fleet_selector.py
Module 2 - Fleet Selection using Genetic Algorithm (GA).

Exact spec formula:  score = 0.75 * coverage_pct - 0.25 * budget_used_pct
Doc note: "A good solution serves most demand without spending the full budget unnecessarily."
Fleet must include BOTH light and heavy drones (min 2 light, min 1 heavy).
"""

import random
from typing import List, Tuple
from grid_model import Cell, get_residential, get_hubs

# ── Drone specs (from project document) ───────────────────────────────────────
DRONE_TYPES = {
    "Light": {"cost": 1000, "payload_kg": 2, "range_cells": 12},
    "Heavy": {"cost": 1800, "payload_kg": 5, "range_cells": 20},
}

BUDGET    = 12000
MIN_LIGHT = 2       # Must have at least 2 light drones
MIN_HEAVY = 1       # Must have at least 1 heavy drone
MAX_TOTAL = 6       # Cap fleet size — keeps budget usage moderate (~60-75%)


def compute_demand(grid: List[List[Cell]]) -> float:
    return sum(grid[r][c].demand for r in range(10) for c in range(10))


def fitness(light: int, heavy: int, grid: List[List[Cell]], budget: float) -> float:
    """
    EXACT spec formula: score = 0.75 * coverage_pct - 0.25 * budget_used_pct

    Hard constraints (return -999 to remove from population):
      - light >= MIN_LIGHT
      - heavy >= MIN_HEAVY
      - cost <= budget
      - total drones <= MAX_TOTAL
    """
    # ── Hard constraints ───────────────────────────────────────────────────────
    if light < MIN_LIGHT or heavy < MIN_HEAVY:
        return -999.0
    cost = light * DRONE_TYPES["Light"]["cost"] + heavy * DRONE_TYPES["Heavy"]["cost"]
    if cost > budget:
        return -999.0
    total = light + heavy
    if total > MAX_TOTAL:
        return -999.0

    # ── Coverage (how many residential zones can the fleet reach) ─────────────
    n_residential = sum(1 for r in range(10) for c in range(10)
                        if grid[r][c].zone == "Residential")
    # Light covers 4 cells per trip, Heavy covers 8 cells per trip
    cells_served  = min(light * 4 + heavy * 8, n_residential)
    coverage_pct  = cells_served / max(n_residential, 1)

    # ── Budget used ────────────────────────────────────────────────────────────
    budget_used_pct = cost / budget

    # ── EXACT SPEC FORMULA ─────────────────────────────────────────────────────
    score = 0.75 * coverage_pct - 0.25 * budget_used_pct

    return round(score, 4)


# ── Genetic Algorithm ─────────────────────────────────────────────────────────

def random_chromosome(budget: float) -> Tuple[int, int]:
    """Random [light, heavy] satisfying all hard constraints."""
    for _ in range(300):
        l = random.randint(MIN_LIGHT, MAX_TOTAL - MIN_HEAVY)
        h = random.randint(MIN_HEAVY, MAX_TOTAL - l)
        if l + h > MAX_TOTAL:
            continue
        cost = l * DRONE_TYPES["Light"]["cost"] + h * DRONE_TYPES["Heavy"]["cost"]
        if cost <= budget:
            return (l, h)
    return (MIN_LIGHT, MIN_HEAVY)


def crossover(p1: Tuple[int, int], p2: Tuple[int, int]):
    """Single-point crossover: swap light or heavy counts."""
    if random.random() < 0.5:
        return (p1[0], p2[1]), (p2[0], p1[1])
    return (p2[0], p1[1]), (p1[0], p2[1])


def mutate(chromosome: Tuple[int, int], budget: float,
           mutation_rate: float = 0.25) -> Tuple[int, int]:
    """±1 mutation on light or heavy. Clamps to minimums and budget."""
    l, h = chromosome
    if random.random() < mutation_rate:
        gene  = random.choice(["light", "heavy"])
        delta = random.choice([-1, 1])
        if gene == "light":
            l = max(MIN_LIGHT, l + delta)
        else:
            h = max(MIN_HEAVY, h + delta)

    # Enforce total cap
    while l + h > MAX_TOTAL:
        if l > MIN_LIGHT:
            l -= 1
        elif h > MIN_HEAVY:
            h -= 1
        else:
            break

    # Enforce budget
    cost = l * DRONE_TYPES["Light"]["cost"] + h * DRONE_TYPES["Heavy"]["cost"]
    while cost > budget:
        if h > MIN_HEAVY:
            h -= 1
        elif l > MIN_LIGHT:
            l -= 1
        else:
            break
        cost = l * DRONE_TYPES["Light"]["cost"] + h * DRONE_TYPES["Heavy"]["cost"]

    return (l, h)


def run_genetic_algorithm(
    grid: List[List[Cell]],
    budget: float = BUDGET,
    population_size: int = 30,
    generations: int = 60,
    mutation_rate: float = 0.25
) -> dict:
    """GA fleet selection. Returns best [light, heavy] under budget."""
    population = [random_chromosome(budget) for _ in range(population_size)]
    best       = (MIN_LIGHT, MIN_HEAVY)
    best_score = -999.0
    history    = []

    for gen in range(generations):
        scored = [(c, fitness(c[0], c[1], grid, budget)) for c in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        gen_best, gen_score = scored[0]
        history.append(gen_score)

        if gen_score > best_score:
            best_score = gen_score
            best       = gen_best

        parents  = [c for c, _ in scored[:max(2, population_size // 2)]]
        next_gen = list(parents)

        while len(next_gen) < population_size:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1, budget, mutation_rate))
            if len(next_gen) < population_size:
                next_gen.append(mutate(c2, budget, mutation_rate))

        population = next_gen[:population_size]

    light, heavy = best
    light      = max(light, MIN_LIGHT)
    heavy      = max(heavy, MIN_HEAVY)
    total_cost = (light * DRONE_TYPES["Light"]["cost"] +
                  heavy * DRONE_TYPES["Heavy"]["cost"])

    return {
        "light_drones":     light,
        "heavy_drones":     heavy,
        "total_drones":     light + heavy,
        "total_cost":       total_cost,
        "budget":           budget,
        "budget_remaining": budget - total_cost,
        "fitness_score":    round(best_score, 4),
        "ga_history":       history,
    }


def print_fleet_report(result: dict):
    print("\n" + "=" * 60)
    print("      AERONET LITE - FLEET SELECTION REPORT (GA)")
    print("=" * 60)
    print(f"  Light Drones  : {result['light_drones']} x $1000 = ${result['light_drones'] * 1000}")
    print(f"  Heavy Drones  : {result['heavy_drones']} x $1800 = ${result['heavy_drones'] * 1800}")
    print(f"  Total Drones  : {result['total_drones']}")
    print(f"  Total Cost    : ${result['total_cost']}")
    print(f"  Budget        : ${result['budget']}")
    print(f"  Remaining     : ${result['budget_remaining']}")
    print(f"  Fitness Score : {result['fitness_score']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    from grid_model import build_grid
    grid   = build_grid()
    result = run_genetic_algorithm(grid)
    print_fleet_report(result)