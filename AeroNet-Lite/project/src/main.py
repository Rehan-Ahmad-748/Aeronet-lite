"""
main.py
AeroNet Lite – Master Integration Script.
Runs all 5 modules end-to-end and produces the 20-step simulation.

Usage:
    python main.py
    python main.py --bike-data path/to/train.csv
"""

import sys
import os
import argparse

# Add src to path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from grid_model import build_grid
from layout_validator import run_validator, print_validation_report
from fleet_selector import run_genetic_algorithm, print_fleet_report
from delivery_simulator import run_simulation
from ml_pipeline import run_demand_forecasting, run_anomaly_detection
# Visualizations are in the Streamlit dashboard: streamlit run src/visualization.py


def main(bike_data_path: str = None):
    print("\n" + "=" * 70)
    print("           AERONET LITE – AUTONOMOUS DRONE DELIVERY SIMULATION")
    print("           BSDS AI Semester Project | SP2026")
    print("=" * 70)

    # ----------------------------------------------------------------
    # MODULE 1 - Build Grid and Validate Layout (CSP)
    # ----------------------------------------------------------------
    print("\n[1/5] MODULE 1 – Grid Build + CSP Layout Validation")
    grid = build_grid()
    report = run_validator(grid)
    print_validation_report(report)

    # ----------------------------------------------------------------
    # MODULE 2 - Fleet Selection (Genetic Algorithm)
    # ----------------------------------------------------------------
    print("[2/5] MODULE 2 – Fleet Selection (Genetic Algorithm)")
    fleet = run_genetic_algorithm(grid, budget=12000, population_size=30, generations=60)
    print_fleet_report(fleet)

    # ----------------------------------------------------------------
    # MODULE 5 - ML Pipeline (done before simulation so results feed into it)
    # ----------------------------------------------------------------
    print("[3/5] MODULE 5 – ML Pipeline (Demand Forecasting + Anomaly Detection)")
    default_bike_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "train.csv")
    )

    if bike_data_path:
        selected_bike_data = bike_data_path
        print(f"[ML] Using --bike-data path: {selected_bike_data}")
    elif os.path.exists(default_bike_csv):
        selected_bike_data = default_bike_csv
        print(f"[ML] Auto-detected Bike Sharing CSV: {selected_bike_data}")
    else:
        selected_bike_data = None
        print("[ML] No real Bike Sharing CSV found. Falling back to synthetic demand data.")

    demand_result = run_demand_forecasting(selected_bike_data)
    anomaly_result = run_anomaly_detection()

    # ----------------------------------------------------------------
    # MODULE 3 & 4 - Simulation with A* + Disruption Handling
    # ----------------------------------------------------------------
    print("\n[4/5] MODULES 3 & 4 – 20-Step Simulation (A* Routing + Disruption Handling)")
    sim_result = run_simulation(grid, fleet, demand_forecast=demand_result)

    # ----------------------------------------------------------------
    # Visualizations
    # ----------------------------------------------------------------
    print("\n[5/5] Streamlit Dashboard ready.")
    

    # ----------------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                    FINAL PROJECT SUMMARY")
    print("=" * 70)
    print(f"  Layout Valid       : {report['valid']}")
    print(f"  Fleet Selected     : {fleet['light_drones']} Light + {fleet['heavy_drones']} Heavy "
          f"(${fleet['total_cost']} / $12000)")
    print(f"  Demand Forecast    : RF MAE={demand_result['models']['RandomForest']['MAE']}, "
          f"RMSE={demand_result['models']['RandomForest']['RMSE']}")
    print(f"  Anomaly Accuracy   : {anomaly_result['models']['RandomForest']['accuracy']:.2%}")
    print(f"  Deliveries Done    : {sim_result['completed']}/{sim_result['total']}")
    print(f"  Delayed / Failed   : {sim_result['delayed']} / {sim_result['failed']}")
    print(f"  Visualization      : streamlit run src/visualization.py")
    print("=" * 70)
    print("\n[DONE] AeroNet Lite simulation complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroNet Lite – Drone Delivery Simulator")
    parser.add_argument("--bike-data", type=str, default=None,
                        help="Path to Bike Sharing train.csv (optional; synthetic if not provided)")
    args = parser.parse_args()
    main(bike_data_path=args.bike_data)