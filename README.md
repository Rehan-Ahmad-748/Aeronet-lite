# AeroNet Lite — Autonomous Drone Delivery & Resilience Simulation

An AI-powered drone delivery simulation platform built as a Data Science project. It optimizes fleet allocation, plans delivery routes, handles real-time disruptions through rerouting, predicts delivery demand, and detects operational anomalies — all visualized through an interactive Streamlit dashboard.

The actual project files are stored under `project/`.

---

## What It Does

- **Route Planning** — uses A* Search to compute optimal delivery paths between nodes
- **Fleet Optimization** — applies Genetic Algorithm (GA) and CSP to allocate drones efficiently
- **Demand Forecasting** — predicts delivery demand using Linear Regression and Random Forest
- **Anomaly Detection** — flags unusual operational patterns using Decision Tree classifier
- **No-Fly Zone Handling** — dynamically reroutes around restricted areas in real time
- **Interactive Dashboard** — built with Streamlit for live monitoring and performance analytics

---

## Tech Stack

`Python` `Scikit-learn` `Streamlit` `Pandas` `NumPy` `A* Search` `Genetic Algorithm`

---

## Project Structure
project/
├── data/         → raw and processed datasets (Kaggle Bike Sharing + synthetic UAV telemetry)
├── src/          → Python source code and simulation modules
├── notebooks/    → Jupyter notebooks for analysis and model training
└── report/       → project report and figures
---

## Getting Started

Open `project/README.md` for detailed setup instructions and steps to run the simulation.

---

*Built as part of BS Data Science coursework at FAST-NUCES Islamabad.*
