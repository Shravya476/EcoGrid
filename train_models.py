"""Trains every model from the CSV files in data/ and saves them in models/.
Run it once with:  python train_models.py   (the app also runs it automatically if a model is missing)."""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")

GEN_FULL_FEATURES = ["hour", "month", "power_2days_ago", "power_3days_ago",
                     "Weather_Temperature_Celsius", "Weather_Relative_Humidity",
                     "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation", "Wind_Speed"]
GEN_WEATHER_FEATURES = ["Weather_Temperature_Celsius", "Weather_Relative_Humidity",      # no hour/month: works at any location
                        "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation", "Wind_Speed"]
DEMAND_FEATURES = ["hour", "day_of_week", "month"]


def load_solar():
    s = pd.read_csv(os.path.join(DATA, "solar_hourly.csv"), parse_dates=["timestamp"], index_col="timestamp")
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    s["hour"], s["month"] = s.index.hour, s.index.month
    s["power_2days_ago"] = s["Active_Power"].shift(48)
    s["power_3days_ago"] = s["Active_Power"].shift(72)
    return s


def load_demand():
    d = pd.read_csv(os.path.join(DATA, "demand_hourly.csv"), parse_dates=["timestamp"], index_col="timestamp")
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h"))
    d["Demand_MW"] = d["Demand_MW"].interpolate(limit_direction="both")
    d["hour"], d["day_of_week"], d["month"] = d.index.hour, d.index.dayofweek, d.index.month
    return d


def _rf():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1)


def _train_generation(s, features, name, label):
    g = s.dropna(subset=features + ["Active_Power"]).copy()
    tr, te = g[g.index < "2015-09-01"], g[g.index >= "2015-09-01"]
    m = _rf().fit(tr[features], tr["Active_Power"])
    pred = np.clip(m.predict(te[features]), 0, None)
    actual = te["Active_Power"].values
    day = te["Global_Horizontal_Radiation"].values > 50
    mae = float(np.abs(actual - pred)[day].mean())
    pct = mae / float(actual[day].mean()) * 100
    last = te.index >= (te.index.max() - pd.Timedelta(days=28))
    low_off, high_off = np.quantile((actual - pred)[last & day], [0.1, 0.9])
    final = _rf().fit(g[features], g["Active_Power"])
    info = {"model": final, "holdout_model": m, "features": features, "low_offset": float(low_off),
            "high_offset": float(high_off), "test_mae_kw": mae, "test_pct": pct, "label": label}
    joblib.dump(info, os.path.join(MODELS, name), compress=3)
    print(f"  {label}: daytime error on unseen months {mae:.2f} kW ({pct:.1f}%)")


def train_generation():
    s = load_solar()
    _train_generation(s, GEN_FULL_FEATURES, "generation_full.joblib", "generation (with recent power)")
    _train_generation(s, GEN_WEATHER_FEATURES, "generation_weather.joblib", "generation (weather only, any location)")


def train_demand():
    d = load_demand()
    tr, te = d[d.index < "2018-07-01"], d[d.index >= "2018-07-01"]
    make = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42)
    m = make().fit(tr[DEMAND_FEATURES], tr["Demand_MW"])
    pred, actual = m.predict(te[DEMAND_FEATURES]), te["Demand_MW"].values
    mape = float(np.mean(np.abs(actual - pred) / actual) * 100)
    last = te.index >= (te.index.max() - pd.Timedelta(days=28))
    low_off, high_off = np.quantile((actual - pred)[last], [0.1, 0.9])
    final = make().fit(d[DEMAND_FEATURES], d["Demand_MW"])
    joblib.dump({"model": final, "holdout_model": m, "features": DEMAND_FEATURES, "low_offset": float(low_off),
                 "high_offset": float(high_off), "test_mape": mape},
                os.path.join(MODELS, "demand_model.joblib"), compress=3)
    print(f"  demand: error on unseen months {mape:.2f}% (MAPE)")


def train_cloud():
    """Optional: needs data/cloud_dataset.npz (made in Colab). Skipped if the file is not there."""
    path = os.path.join(DATA, "cloud_dataset.npz")
    if not os.path.exists(path):
        print("  cloud models: skipped (data/cloud_dataset.npz not found)")
        return
    import cloud_features
    z = np.load(path, allow_pickle=True)
    X, Y, NOW = z["X"], z["Y"], z["NOW"]
    WHEN = pd.DatetimeIndex(z["WHEN"])
    make = lambda: HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05,
                                                 max_iter=150, l2_regularization=1.0, random_state=42)
    F = np.array([cloud_features.example_features(X[i], NOW[i], WHEN[i].hour, WHEN[i].month) for i in range(len(X))])
    models = [make().fit(F, Y[:, k] - NOW) for k in range(3)]
    joblib.dump({"models": models, "leads_hours": [1, 2, 3]}, os.path.join(MODELS, "cloud_models.joblib"))
    E = np.array([cloud_features.estimate_features(X[i, 2], WHEN[i].hour, WHEN[i].month) for i in range(len(X))])
    joblib.dump(make().fit(E, NOW), os.path.join(MODELS, "cloud_now_model.joblib"))
    print("  cloud models: trained")


def ensure_models(force=False):
    os.makedirs(MODELS, exist_ok=True)
    need = lambda f: force or not os.path.exists(os.path.join(MODELS, f))
    if need("generation_full.joblib") or need("generation_weather.joblib"):
        print("Training the generation models (about a minute)...")
        train_generation()
    if need("demand_model.joblib"):
        print("Training the demand model...")
        train_demand()
    if need("cloud_models.joblib") or need("cloud_now_model.joblib"):
        print("Training the cloud models...")
        train_cloud()


if __name__ == "__main__":
    ensure_models(force=True)
    print("Done. Models saved in", MODELS)
