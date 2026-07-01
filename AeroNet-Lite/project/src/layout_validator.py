"""
layout_validator.py
Module 1 - Layout Validation using Constraint Satisfaction Problem (CSP).
Checks four city layout rules and reports violations with fix suggestions.
"""

from grid_model import (
    Cell, get_neighbors, manhattan,
    get_hubs, get_charging_pads, get_medical_pickups,
    get_hospitals, get_residential, get_industrial
)
from typing import List


def check_industrial_safety(grid: List[List[Cell]]) -> list:
    """
    Rule R1: Industrial cells cannot be directly adjacent to Schools or Hospitals.
    """
    errors = []
    sensitive_zones = {"School", "Hospital"}
    for (r, c) in get_industrial(grid):
        for (nr, nc) in get_neighbors(r, c):
            neighbor_zone = grid[nr][nc].zone
            if neighbor_zone in sensitive_zones:
                errors.append({
                    "rule": "R1",
                    "cell": (r, c),
                    "detail": f"Industrial cell ({r},{c}) is adjacent to {neighbor_zone} cell ({nr},{nc}).",
                    "fix": f"Add a buffer zone (Open Field or Commercial) between ({r},{c}) and ({nr},{nc})."
                })
    return errors


def check_residential_coverage(grid: List[List[Cell]]) -> list:
    """
    Rule R2: Every Residential cell must be within 3 Manhattan cells of a Drone Hub.
    """
    errors = []
    hubs = get_hubs(grid)
    for (r, c) in get_residential(grid):
        min_dist = min((manhattan((r, c), h) for h in hubs), default=999)
        if min_dist > 3:
            nearest_hub = min(hubs, key=lambda h: manhattan((r, c), h)) if hubs else None
            fix_r = (r + nearest_hub[0]) // 2 if nearest_hub else r
            fix_c = (c + nearest_hub[1]) // 2 if nearest_hub else c
            errors.append({
                "rule": "R2",
                "cell": (r, c),
                "detail": f"Residential cell ({r},{c}) is {min_dist} cells from the nearest hub (max allowed: 3).",
                "fix": f"Add a hub near ({fix_r},{fix_c}) or convert ({r},{c}) to Open Field."
            })
    return errors


def check_hub_charging(grid: List[List[Cell]]) -> list:
    """
    Rule R3: Every Drone Hub must have a Charging Pad within 2 Manhattan cells.
    """
    errors = []
    charging_pads = get_charging_pads(grid)
    for (r, c) in get_hubs(grid):
        min_dist = min((manhattan((r, c), p) for p in charging_pads), default=999)
        if min_dist > 2:
            errors.append({
                "rule": "R3",
                "cell": (r, c),
                "detail": f"Hub ({r},{c}) has no charging pad within 2 cells (nearest is {min_dist} away).",
                "fix": f"Place a charging pad at ({r},{min(c+1,9)}) or ({min(r+1,9)},{c})."
            })
    return errors


def check_medical_access(grid: List[List[Cell]]) -> list:
    """
    Rule R4: At least one Hospital must have a Medical Pickup point within 1 cell.
    """
    errors = []
    hospitals = get_hospitals(grid)
    medical_pickups = get_medical_pickups(grid)

    covered = False
    for (hr, hc) in hospitals:
        for (mr, mc) in medical_pickups:
            if manhattan((hr, hc), (mr, mc)) <= 1:
                covered = True
                break
        if covered:
            break

    if not covered:
        if hospitals:
            hr, hc = hospitals[0]
            errors.append({
                "rule": "R4",
                "cell": "All Hospitals",
                "detail": "No hospital has a Medical Pickup point within 1 cell.",
                "fix": f"Add a Medical Pickup at ({hr},{min(hc+1,9)}) adjacent to hospital ({hr},{hc})."
            })
    return errors


def run_validator(grid: List[List[Cell]]) -> dict:
    """
    Run all CSP validation rules and return a structured report.
    """
    all_errors = []
    rule_results = {}

    checks = [
        ("R1 - Industrial Safety",     check_industrial_safety),
        ("R2 - Residential Coverage",  check_residential_coverage),
        ("R3 - Hub Charging",          check_hub_charging),
        ("R4 - Medical Access",        check_medical_access),
    ]

    for rule_name, fn in checks:
        errs = fn(grid)
        rule_results[rule_name] = errs
        all_errors.extend(errs)

    passed = [name for name, errs in rule_results.items() if len(errs) == 0]
    failed = [name for name, errs in rule_results.items() if len(errs) > 0]
    is_valid = len(all_errors) == 0

    return {
        "valid": is_valid,
        "passed": passed,
        "failed": failed,
        "errors": all_errors,
        "rule_results": rule_results
    }


def print_validation_report(report: dict):
    print("\n" + "="*60)
    print("      AERONET LITE - CSP LAYOUT VALIDATION REPORT")
    print("="*60)
    validity = "PASSED" if report["valid"] else "FAILED"
    print(f"\nLayout Validity: {validity}")
    print(f"Rules Passed   : {len(report['passed'])}/4")
    print(f"Rules Failed   : {len(report['failed'])}/4")

    if report["passed"]:
        print("\n[PASSED RULES]")
        for r in report["passed"]:
            print(f"  [PASS] {r}")

    if report["failed"]:
        print("\n[FAILED RULES]")
        for r in report["failed"]:
            print(f"  [FAIL] {r}")
            for err in report["rule_results"][r]:
                print(f"     ERROR  : {err['detail']}")
                print(f"     FIX    : {err['fix']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    from grid_model import build_grid
    grid = build_grid()
    report = run_validator(grid)
    print_validation_report(report)
