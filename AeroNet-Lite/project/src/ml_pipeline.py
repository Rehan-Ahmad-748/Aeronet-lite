"""
ml_pipeline.py
Module 5 - Demand Forecasting (Regression) and Anomaly Detection (Classification).
Uses Bike Sharing Dataset for demand forecasting.
Uses synthetic drone telemetry for anomaly detection.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PART 1 - DEMAND FORECASTING
# ============================================================

def load_bike_sharing(data_path: str = None) -> pd.DataFrame:
    """
    Load the Bike Sharing dataset.
    If the CSV is not found, generate synthetic data with the same structure.
    """
    default_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "train.csv")
    )

    candidate_path = None
    if data_path:
        candidate_path = data_path
    elif os.path.exists(default_csv):
        candidate_path = default_csv

    if candidate_path and os.path.exists(candidate_path):
        df = pd.read_csv(candidate_path)
        print(f"[ML] Using real Bike Sharing data: {candidate_path} | shape={df.shape}")
    else:
        print("[ML] Bike Sharing CSV not found. Generating synthetic data with same structure.")
        np.random.seed(42)
        n = 10886
        hours     = np.tile(np.arange(24), n // 24 + 1)[:n]
        seasons   = np.random.randint(1, 5, n)
        holidays  = np.random.randint(0, 2, n)
        workdays  = np.random.randint(0, 2, n)
        weather   = np.random.randint(1, 5, n)
        temp      = np.random.uniform(0.1, 0.9, n)
        atemp     = temp + np.random.uniform(-0.05, 0.05, n)
        humidity  = np.random.uniform(0.2, 0.9, n)
        windspeed = np.random.uniform(0.0, 0.5, n)

        # Count = base + time-of-day peak + random noise
        base  = 100 + 200 * temp + 50 * workdays
        peak  = 150 * np.sin(np.pi * hours / 24)
        count = (base + peak + np.random.normal(0, 30, n)).clip(1).astype(int)

        df = pd.DataFrame({
            "season": seasons, "holiday": holidays, "workingday": workdays,
            "weather": weather, "temp": temp.round(4), "atemp": atemp.round(4),
            "humidity": humidity.round(4), "windspeed": windspeed.round(4),
            "hour": hours, "count": count
        })
    return df


def run_demand_forecasting(data_path: str = None) -> dict:
    """
    Train Linear Regression and Random Forest Regressor on bike sharing data.
    Returns MAE, RMSE, and a sample predicted demand for grid scaling.
    """
    df = load_bike_sharing(data_path)

    # Feature selection
    features = ["season", "holiday", "workingday", "weather",
                "temp", "atemp", "humidity", "windspeed", "hour"]
    # Keep only columns that exist
    features = [f for f in features if f in df.columns]
    target = "count"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # ---- Linear Regression ----
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["LinearRegression"] = {
        "MAE":  round(mean_absolute_error(y_test, y_pred_lr), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 2)
    }

    # ---- Random Forest Regressor ----
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["RandomForest"] = {
        "MAE":  round(mean_absolute_error(y_test, y_pred_rf), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 2)
    }

    # Sample prediction for grid scaling (hour=10, workday, mild weather)
    sample = pd.DataFrame([{
        "season": 2, "holiday": 0, "workingday": 1,
        "weather": 1, "temp": 0.5, "atemp": 0.5,
        "humidity": 0.5, "windspeed": 0.2, "hour": 10
    }])
    sample = sample[[f for f in features if f in sample.columns]]
    predicted_demand = float(rf.predict(sample)[0])
    avg_demand = float(y_pred_rf.mean())

    print("\n[ML] Demand Forecasting Results:")
    for model, metrics in results.items():
        print(f"  {model}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}")

    processed_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    )
    os.makedirs(processed_dir, exist_ok=True)
    predictions_path = os.path.join(processed_dir, "demand_predictions.csv")

    pred_df = pd.DataFrame({
        "actual_count": y_test.values,
        "predicted_count": np.round(y_pred_rf, 2),
        "model": "RandomForest"
    })
    pred_df.to_csv(predictions_path, index=False)
    print(f"[ML] Saved RandomForest demand predictions to: {predictions_path}")

    return {
        "models": results,
        "predicted_demand": round(predicted_demand, 2),
        "avg_demand": round(avg_demand, 2),
        "best_model": "RandomForest",
        "predictions_csv": predictions_path
    }


# ============================================================
# PART 2 - ANOMALY DETECTION
# ============================================================

def generate_synthetic_telemetry(n_normal: int = 500, n_anomaly: int = 150) -> pd.DataFrame:
    """
    Overlapping ranges + Gaussian noise so accuracy is realistic (~85-95%).
    """
    np.random.seed(42)
    records = []

    def noisy(val, std=1.5):
        return val + np.random.normal(0, std)

    # Normal
    for _ in range(n_normal):
        records.append({
            "battery_drop":    max(0.5, noisy(np.random.uniform(1.0, 5.0), 1.0)),
            "speed":           max(1.0, noisy(np.random.uniform(8.0, 14.0), 1.5)),
            "route_deviation": max(0.0, noisy(np.random.uniform(0.0, 1.0), 0.3)),
            "altitude_change": noisy(np.random.uniform(-1.0, 1.0), 0.5),
            "speed_change":    noisy(np.random.uniform(-2.0, 2.0), 0.8),
            "label": "Normal"
        })

    # Battery Anomaly — overlaps with Normal at low battery_drop values
    for _ in range(n_anomaly):
        records.append({
            "battery_drop":    max(0.5, noisy(np.random.uniform(8.0, 20.0), 3.0)),
            "speed":           max(1.0, noisy(np.random.uniform(6.0, 13.0), 2.0)),
            "route_deviation": max(0.0, noisy(np.random.uniform(0.0, 1.5), 0.5)),
            "altitude_change": noisy(np.random.uniform(-1.5, 1.5), 0.8),
            "speed_change":    noisy(np.random.uniform(-3.0, 3.0), 1.0),
            "label": "Battery Anomaly"
        })

    # Route Anomaly — overlaps with Normal at low deviation values
    for _ in range(n_anomaly):
        records.append({
            "battery_drop":    max(0.5, noisy(np.random.uniform(1.0, 6.0), 1.5)),
            "speed":           max(1.0, noisy(np.random.uniform(7.0, 15.0), 2.0)),
            "route_deviation": max(0.0, noisy(np.random.uniform(2.0, 5.0), 1.0)),
            "altitude_change": noisy(np.random.uniform(-2.0, 2.0), 1.0),
            "speed_change":    noisy(np.random.uniform(-3.0, 3.0), 1.2),
            "label": "Route Anomaly"
        })

    # Sensor Spike — overlaps with Normal at low altitude/speed_change values
    for _ in range(n_anomaly):
        records.append({
            "battery_drop":    max(0.5, noisy(np.random.uniform(1.0, 6.0), 1.5)),
            "speed":           max(1.0, noisy(np.random.uniform(7.0, 16.0), 2.5)),
            "route_deviation": max(0.0, noisy(np.random.uniform(0.0, 2.0), 0.8)),
            "altitude_change": noisy(np.random.uniform(4.0, 12.0), 2.5),
            "speed_change":    noisy(np.random.uniform(5.0, 15.0), 3.0),
            "label": "Sensor Spike"
        })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def run_anomaly_detection() -> dict:
    """
    Train Decision Tree and Random Forest classifiers on synthetic telemetry.
    Returns accuracy, classification report, confusion matrix.
    """
    df = generate_synthetic_telemetry()
    print(f"\n[ML] Anomaly dataset: {df.shape}, class distribution:\n{df['label'].value_counts().to_string()}")

    features = ["battery_drop", "speed", "route_deviation", "altitude_change", "speed_change"]
    X = df[features]

    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # ---- Decision Tree ----
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results["DecisionTree"] = {
        "accuracy": round(accuracy_score(y_test, y_pred_dt), 4),
        "report": classification_report(y_test, y_pred_dt,
                                        target_names=le.classes_, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_dt).tolist()
    }

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["RandomForest"] = {
        "accuracy": round(accuracy_score(y_test, y_pred_rf), 4),
        "report": classification_report(y_test, y_pred_rf,
                                        target_names=le.classes_, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
    }

    print("\n[ML] Anomaly Detection Results:")
    for model, res in results.items():
        print(f"  {model}: Accuracy={res['accuracy']:.4f}")
    print("\n  Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    print("  Confusion Matrix:")
    print(np.array(results["RandomForest"]["confusion_matrix"]))

    return {
        "models": results,
        "classes": list(le.classes_),
        "best_model": "RandomForest",
        "label_encoder": le,
        "rf_classifier": rf,
        "features": features
    }


def predict_anomaly(telemetry: dict, rf_model, features: list, le: LabelEncoder) -> str:
    """
    Predict anomaly class for a single drone telemetry reading.
    """
    row = pd.DataFrame([{f: telemetry.get(f, 0.0) for f in features}])
    pred = rf_model.predict(row)[0]
    return le.inverse_transform([pred])[0]


if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 5 - ML PIPELINE")
    print("=" * 60)
    demand_result = run_demand_forecasting()
    anomaly_result = run_anomaly_detection()

    print("\n[Summary]")
    print(f"  Best Demand Model : {demand_result['best_model']}")
    print(f"  Predicted Demand  : {demand_result['predicted_demand']}")
    print(f"  Best Anomaly Model: {anomaly_result['best_model']}")
    print(f"  Anomaly Accuracy  : {anomaly_result['models']['RandomForest']['accuracy']}")
