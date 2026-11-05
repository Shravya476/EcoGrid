"""EcoGrid AI - solar generation and energy consumption forecasting app (Flask)."""
import os, io, json, base64, sqlite3, urllib.request, urllib.parse, re, secrets, hashlib, smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import train_models
import cloud_live

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("ECOGRID_DB_PATH", os.path.join(HERE, "users.db"))
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ecogrid-fresh-session-2026-10-27")
# New cookie name intentionally invalidates any old browser login session from earlier builds.
app.config["SESSION_COOKIE_NAME"] = "ecogrid_session_v2"

# ---------------- Settings ----------------
REF_LAT, REF_LON = -23.76, 133.87         # approximate location of the reference solar plant (Alice Springs)
LOCAL_OFFSET = timedelta(hours=9, minutes=30)     # Alice Springs time (used for the reference plant's own data)
CAPACITY_KWH, MAX_POWER_KW, EFFICIENCY, MIN_SOC = 60.0, 15.0, 0.95, 0.10      # battery assumptions
DEMAND_SHARE = 0.60
LOW_OUTPUT = 0.60
SUN_WEAKER, SUN_STRONGER = 0.15, 0.15     # forecast uncertainty band (assumption, not measured plant error)
DEMO_START = pd.Timestamp("2015-09-04")   # first date the generation model never saw (plus 3 days of history)
DEMAND_DEMO_START = pd.Timestamp("2018-07-01")

# ---------------- Data and models ----------------
train_models.ensure_models()
SOLAR = train_models.load_solar()
DEMAND = train_models.load_demand()
DEM = joblib.load(os.path.join(train_models.MODELS, "demand_model.joblib"))
SCALE = DEMAND_SHARE * SOLAR["Active_Power"].mean() / DEMAND["Demand_MW"].mean()      # MW -> plant kW
REF_PEAK = round(float(SOLAR["Active_Power"].max()), 1)                                # reference plant's highest recorded output (kW)
DEFAULT_LOC = {"name": "Reference plant, Alice Springs (Australia)", "lat": REF_LAT, "lon": REF_LON,
               "kw": REF_PEAK, "reference": True, "offset": int(LOCAL_OFFSET.total_seconds()),
               "plant_type": "monocrystalline", "mounting": "fixed",
               "plant_status": "existing", "plant_type_known": True, "mounting_known": True}


def recommend_capacity(monthly_bill, available_area=None, purpose="home"):
    """Give a simple planning estimate when the user does not know plant kW.

    This is deliberately a planning estimate, not a quotation or engineering design.
    It converts an approximate monthly electricity bill to energy use, then assumes
    about 3.8 peak-sun-hours/day and 80% overall system performance.
    """
    try:
        bill = max(float(monthly_bill), 0.0)
    except (TypeError, ValueError):
        bill = 0.0
    # Simple Indian residential/commercial planning assumption: roughly Rs 8/kWh.
    monthly_kwh = bill / 8.0 if bill > 0 else 300.0
    required = monthly_kwh / (30.0 * 3.8 * 0.80)
    required = max(1.0, min(required, 100000.0))
    if available_area not in (None, ""):
        try:
            area = max(float(available_area), 0.0)
            if area > 0:
                area_limited = area / 9.0  # approx. 9 m²/kW planning allowance
                if area_limited < required:
                    required = max(1.0, area_limited)
        except (TypeError, ValueError):
            pass
    # Round to practical half-kW increments.
    return round(max(1.0, required) * 2) / 2.0


def bill_based_consumption(monthly_bill=None, monthly_units=None, tariff=8.0):
    """Estimate average daily electricity use from a bill or directly reported kWh.

    Directly reported monthly kWh is preferred. If unavailable, the bill is divided
    by an editable approximate tariff. This is an estimate because electricity
    tariffs vary by slab, provider and fixed charges.
    """
    try:
        units = float(monthly_units) if monthly_units not in (None, "") else None
    except (TypeError, ValueError):
        units = None
    if units is not None and 0 <= units <= 10000000:
        return {"monthly_kwh": units, "daily_kwh": units / 30.0, "source": "monthly units from electricity bill", "tariff": None}

    try:
        bill = float(monthly_bill) if monthly_bill not in (None, "") else None
    except (TypeError, ValueError):
        bill = None
    try:
        rate = float(tariff)
    except (TypeError, ValueError):
        rate = 8.0
    rate = min(1000.0, max(0.01, rate))
    if bill is not None and 0 <= bill <= 100000000:
        units = bill / rate
        return {"monthly_kwh": units, "daily_kwh": units / 30.0, "source": f"monthly bill using approximate ₹{rate:g}/kWh tariff", "tariff": rate}
    return None


# ---------------- Users (SQLite) ----------------
def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 30000")
    return con


def init_db():
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)")
        # Add email to existing databases without deleting or resetting accounts.
        user_columns = {row[1] for row in con.execute("PRAGMA table_info(users)").fetchall()}
        if "email" not in user_columns:
            con.execute("ALTER TABLE users ADD COLUMN email TEXT")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(lower(email)) WHERE email IS NOT NULL AND email != ''")
        con.execute("""CREATE TABLE IF NOT EXISTS pending_registrations (
            username TEXT PRIMARY KEY, email TEXT NOT NULL, password_hash TEXT NOT NULL,
            otp_hash TEXT NOT NULL, expires_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL)""")
        # One-time fresh-start migration. Older EcoGrid builds could contain a
        # pre-created demo account (such as Sravya). This clears those accounts
        # exactly once for this release, then records the migration so real users
        # created afterwards are not deleted on normal restarts.
        con.execute("CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS user_locations (username TEXT PRIMARY KEY, name TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL, updated_at TEXT NOT NULL)")
        con.execute("""CREATE TABLE IF NOT EXISTS user_settings (
            username TEXT PRIMARY KEY, plant_kw REAL NOT NULL DEFAULT 5,
            plant_type TEXT NOT NULL DEFAULT 'monocrystalline',
            mounting TEXT NOT NULL DEFAULT 'fixed', plant_status TEXT NOT NULL DEFAULT 'existing',
            purpose TEXT, monthly_bill REAL, monthly_units REAL, tariff REAL DEFAULT 8,
            available_area REAL, updated_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS consumption_profiles (
            username TEXT PRIMARY KEY, monthly_bill REAL, monthly_units REAL,
            tariff REAL DEFAULT 8, system_kw REAL DEFAULT 5, appliance_plan_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS forecast_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL,
            forecast_type TEXT NOT NULL, location_name TEXT, latitude REAL, longitude REAL,
            forecast_date TEXT NOT NULL, horizon_hours INTEGER, generation_kwh REAL,
            consumption_kwh REAL, energy_balance_kwh REAL, capacity_kw REAL,
            details_json TEXT, created_at TEXT NOT NULL)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_forecast_user_date ON forecast_history(username, forecast_date)")
        row = con.execute("SELECT value FROM app_meta WHERE key = 'user_database_version'").fetchone()
        if not row or row[0] != "fresh-users-v2":
            con.execute("DELETE FROM users")
            con.execute("INSERT OR REPLACE INTO app_meta (key, value) VALUES ('user_database_version', 'fresh-users-v2')")
            con.commit()


init_db()


def persist_plant_settings(username, loc):
    """Save the user's solar plant and planning inputs independently of the browser session."""
    now = datetime.now(timezone.utc).isoformat()
    with db() as con:
        con.execute("""INSERT INTO user_settings
            (username,plant_kw,plant_type,mounting,plant_status,purpose,monthly_bill,monthly_units,tariff,available_area,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(username) DO UPDATE SET
            plant_kw=excluded.plant_kw, plant_type=excluded.plant_type, mounting=excluded.mounting,
            plant_status=excluded.plant_status, purpose=excluded.purpose, monthly_bill=excluded.monthly_bill,
            monthly_units=excluded.monthly_units, tariff=excluded.tariff, available_area=excluded.available_area,
            updated_at=excluded.updated_at""",
            (username, float(loc.get("kw", 5)), loc.get("plant_type", "monocrystalline"),
             loc.get("mounting", "fixed"), loc.get("plant_status", "existing"),
             loc.get("purpose"), loc.get("monthly_bill"), loc.get("monthly_units"),
             float(loc.get("tariff", 8) or 8), loc.get("available_area"), now))


def load_plant_settings(username):
    with db() as con:
        row = con.execute("SELECT * FROM user_settings WHERE username=?", (username,)).fetchone()
    if not row:
        return None
    data = dict(row)
    return {"kw": data["plant_kw"], "plant_type": data["plant_type"], "mounting": data["mounting"],
            "plant_status": data["plant_status"], "purpose": data["purpose"],
            "monthly_bill": data["monthly_bill"], "monthly_units": data["monthly_units"],
            "tariff": data["tariff"], "available_area": data["available_area"]}


def save_forecast(username, forecast_type, loc, forecast_date, horizon, generation,
                  consumption=None, capacity=None, details=None):
    now = datetime.now(timezone.utc).isoformat()
    with db() as con:
        con.execute("""INSERT INTO forecast_history
            (username,forecast_type,location_name,latitude,longitude,forecast_date,horizon_hours,
             generation_kwh,consumption_kwh,energy_balance_kwh,capacity_kw,details_json,created_at)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (username, forecast_type, loc.get("name"), loc.get("lat"), loc.get("lon"),
             str(forecast_date), int(horizon), generation, consumption,
             (generation-consumption) if generation is not None and consumption is not None else None,
             capacity, json.dumps(details or {}, default=str), now))


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login"))
        return view(*a, **kw)
    return wrapped


# ---------------- Forecast logic ----------------
def local_today():
    return (datetime.now(timezone.utc) + LOCAL_OFFSET).date()


def hours_of(day):
    return pd.date_range(pd.Timestamp(day), periods=24, freq="h")


def fetch_weather_forecast(lat, lon):
    """Fetch current + hourly 3-day weather forecast in the location's local timezone.
    The generation engine uses forecast radiation/weather directly; it does not require
    historical generation from the user's plant."""
    hourly_fields = (
        "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,"
        "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,precipitation"
    )
    # Use Open-Meteo Best Match for the closest suitable model for the exact
    # location. This is preferable to forcing one global model in a local
    # weather screen, because Best Match can select higher-resolution models
    # where they are available. It remains completely free and needs no key.
    base_params = {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
        "cell_selection": "nearest",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,cloud_cover,precipitation,weather_code,shortwave_radiation",
        "hourly": hourly_fields + ",weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,sunrise,sunset"
    }
    params = urllib.parse.urlencode(base_params)
    try:
        with urllib.request.urlopen("https://api.open-meteo.com/v1/forecast?" + params, timeout=20) as r:
            data = json.load(r)
    except Exception:
        # Keep the app usable if the primary forecast endpoint is temporarily
        # unavailable by retrying the same free Best Match request.
        params = urllib.parse.urlencode(base_params)
        with urllib.request.urlopen("https://api.open-meteo.com/v1/forecast?" + params, timeout=20) as r:
            data = json.load(r)
    h = data["hourly"]
    idx = pd.to_datetime(h["time"])
    df = pd.DataFrame({
        "temperature": np.asarray(h["temperature_2m"], dtype=float),
        "humidity": np.asarray(h["relative_humidity_2m"], dtype=float),
        "wind_ms": np.asarray(h["wind_speed_10m"], dtype=float) / 3.6,
        "cloud_cover": np.asarray(h["cloud_cover"], dtype=float),
        "ghi": np.asarray(h["shortwave_radiation"], dtype=float),
        "dni": np.asarray(h["direct_normal_irradiance"], dtype=float),
        "diffuse": np.asarray(h["diffuse_radiation"], dtype=float),
        "precipitation": np.asarray(h["precipitation"], dtype=float),
    }, index=idx)
    daily = data.get("daily", {}) or {}
    current = data.get("current", {}) or {}
    # Open-Meteo returns hourly values from the start of the local day.
    # The current object is the actual current forecast hour, so the Home page
    # must not use hourly[0] as if it were "now".
    daily_out = {
        "time": daily.get("time", []),
        "weather_code": daily.get("weather_code", []),
        "temperature_max": daily.get("temperature_2m_max", []),
        "temperature_min": daily.get("temperature_2m_min", []),
        "precipitation_sum": daily.get("precipitation_sum", []),
        "precipitation_probability_max": daily.get("precipitation_probability_max", []),
        "sunrise": daily.get("sunrise", []),
        "sunset": daily.get("sunset", []),
    }
    current["timezone"] = data.get("timezone", "")
    current["weather_source"] = "Open-Meteo Best Match"
    current["utc_offset_seconds"] = int(data.get("utc_offset_seconds", 0))
    current["requested_latitude"] = float(lat)
    current["requested_longitude"] = float(lon)
    current["resolved_latitude"] = float(data.get("latitude", lat))
    current["resolved_longitude"] = float(data.get("longitude", lon))
    current["weather_generated_at"] = current.get("time", "")
    current["hourly_times"] = [str(x) for x in h["time"]]
    current["hourly_temperature"] = [float(x) for x in h["temperature_2m"]]
    current["hourly_cloud"] = [float(x) for x in h["cloud_cover"]]
    current["hourly_wind"] = [float(x) for x in h["wind_speed_10m"]]
    current["hourly_precipitation"] = [float(x) for x in h["precipitation"]]
    current["hourly_weather_code"] = [int(x) for x in h.get("weather_code", [0] * len(h["time"]))]
    current["daily"] = daily_out
    return df, int(data.get("utc_offset_seconds", 0)), current


def search_places(query):
    """Place name -> list of matches (Open-Meteo geocoding, free, no key)."""
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": query, "count": 6, "language": "en", "format": "json"})
    with urllib.request.urlopen(url, timeout=15) as r:
        results = json.load(r).get("results") or []
    return [{"name": p.get("name", ""), "region": p.get("admin1", ""), "country": p.get("country", ""),
             "lat": p["latitude"], "lon": p["longitude"]} for p in results]


def demand_for(day):
    day = pd.Timestamp(day)
    rows = hours_of(day)
    feats = pd.DataFrame({"hour": rows.hour, "day_of_week": rows.dayofweek, "month": rows.month}, index=rows)
    seen_in_history = rows[0] in DEMAND.index and rows[-1] in DEMAND.index
    unseen = seen_in_history and day >= DEMAND_DEMO_START
    mw = (DEM["holdout_model"] if unseen else DEM["model"]).predict(feats[DEM["features"]])
    out = {"expected": mw * SCALE, "low": (mw + DEM["low_offset"]) * SCALE, "high": (mw + DEM["high_offset"]) * SCALE,
           "actual": None}
    if unseen:
        out["actual"] = DEMAND.loc[rows, "Demand_MW"].values * SCALE
        out["mode"] = "Historical demo: the model never saw these dates, and the forecast is compared with the recorded demand."
    elif seen_in_history:
        out["actual"] = DEMAND.loc[rows, "Demand_MW"].values * SCALE
        out["mode"] = "This date was part of the model's training data, so a close match with the recorded demand is NOT a test. For a fair test pick a date between 2018-07-01 and 2018-12-31."
    else:
        out["mode"] = "Forecast for a date with no recorded demand data."
    return out


PLANT_TYPES = {
    "monocrystalline": {"label": "Monocrystalline", "performance": 0.98, "temp_coeff": -0.0035},
    "polycrystalline": {"label": "Polycrystalline", "performance": 0.94, "temp_coeff": -0.0040},
    "thin-film": {"label": "Thin-film", "performance": 0.88, "temp_coeff": -0.0020},
}
MOUNTING_TYPES = {
    "fixed": {"label": "Fixed tilt", "gain": 1.00},
    "single-axis": {"label": "Single-axis tracking", "gain": 1.08},
    "dual-axis": {"label": "Dual-axis tracking", "gain": 1.12},
}

def _weather_situation(row):
    cloud = float(row.get("cloud_cover", 0) or 0)
    rain = float(row.get("precipitation", 0) or 0)
    if rain >= 1.0:
        return "Rain / very cloudy"
    if cloud >= 80:
        return "Overcast"
    if cloud >= 50:
        return "Cloudy"
    if cloud >= 20:
        return "Partly cloudy"
    return "Mostly clear"


def _select_forecast_window(wf, horizon, current_time=None):
    # Start at the API's current local hour, then take the requested 24/48 forecast hours.
    now = pd.Timestamp(current_time).floor("h") if current_time else pd.Timestamp.now(tz=None).floor("h")
    candidates = wf.index[wf.index >= now]
    if len(candidates) == 0:
        candidates = wf.index
    return wf.loc[candidates[:horizon]].copy()


def generation_for(day, loc, horizon=24):
    """Physics-informed solar forecast for a new plant with no historical generation.

    Nameplate capacity is the user's installed kW. Forecast GHI is converted to a
    capacity fraction, then adjusted for module type, mounting and temperature.
    This avoids pretending that the Alice Springs reference model is valid everywhere.
    """
    horizon = 48 if int(horizon) >= 48 else 24
    try:
        wf, offset, current = fetch_weather_forecast(loc["lat"], loc["lon"])
    except Exception as e:
        raise ValueError("Could not reach the weather forecast service (" + str(e)[:100] + "). Check your Internet connection.")
    loc["offset"] = offset
    sub = _select_forecast_window(wf, horizon, current.get("time"))
    if len(sub) < horizon:
        raise ValueError("The weather service did not return enough forecast hours. Please try again in a moment.")

    plant_type = loc.get("plant_type", "monocrystalline")
    mounting = loc.get("mounting", "fixed")
    pt = PLANT_TYPES.get(plant_type, PLANT_TYPES["monocrystalline"])
    mt = MOUNTING_TYPES.get(mounting, MOUNTING_TYPES["fixed"])
    capacity = float(loc.get("kw", 5.0))

    ghi = np.maximum(sub["ghi"].to_numpy(dtype=float), 0)
    temp = sub["temperature"].to_numpy(dtype=float)
    # PV output is approximately proportional to incident irradiance. At 1000 W/m²
    # a nameplate-rated system is near its rated DC output under standard conditions.
    irradiance_factor = np.clip(ghi / 1000.0, 0, 1.0)
    temperature_factor = 1.0 + pt["temp_coeff"] * np.maximum(temp - 25.0, 0)
    temperature_factor = np.clip(temperature_factor, 0.75, 1.05)
    expected = capacity * irradiance_factor * pt["performance"] * mt["gain"] * temperature_factor
    expected = np.clip(expected, 0, capacity)
    expected[ghi < 5] = 0

    # Forecast range represents weather uncertainty, not historical model error.
    low = np.clip(expected * (1 - SUN_WEAKER), 0, capacity)
    high = np.clip(expected * (1 + SUN_STRONGER), 0, capacity)
    low[ghi < 5] = 0
    high[ghi < 5] = 0

    energy = float(expected.sum())
    low_energy = float(low.sum())
    high_energy = float(high.sum())
    peak_i = int(np.argmax(expected))
    clear_hours = int((sub["cloud_cover"] < 20).sum())
    cloudy_hours = int((sub["cloud_cover"] >= 70).sum())
    rainy_hours = int((sub["precipitation"] >= 1.0).sum())
    situation_counts = sub.apply(_weather_situation, axis=1).value_counts()
    situation = str(situation_counts.index[0]) if len(situation_counts) else "Unknown"

    mode = ("Live 24/48-hour forecast: current location + Open-Meteo weather + installed plant capacity. "
            "No historical generation data from this plant is required.")
    band_note = ("Range is an engineering uncertainty band of ±15% around the weather-driven estimate. "
                 "It is an assumption, not an accuracy score from this plant's historical readings.")
    return {
        "expected": expected, "low": low, "high": high, "actual": None,
        "mode": mode, "band_note": band_note, "test_pct": None, "live": True,
        "factor": 1.0, "horizon": horizon, "timestamps": sub.index,
        "weather": sub, "energy": energy, "low_energy": low_energy, "high_energy": high_energy,
        "peak_hour": peak_i, "peak": float(expected[peak_i]),
        "situation": situation, "clear_hours": clear_hours, "cloudy_hours": cloudy_hours,
        "rainy_hours": rainy_hours, "avg_cloud": float(sub["cloud_cover"].mean()),
        "avg_temp": float(sub["temperature"].mean()), "current": current,
        "plant_type_label": pt["label"], "mounting_label": mt["label"],
        "capacity": capacity,
    }


def simulate(gen, dem, factor=1.0, soc_start=0.5):
    cap, pmax = CAPACITY_KWH * factor, MAX_POWER_KW * factor              # battery scales with plant size
    soc, rows = soc_start * cap, []
    for g, d in zip(gen, dem):
        backup = 0.0
        if g >= d:
            charge = min(g - d, pmax, (cap - soc) / EFFICIENCY)
            soc += charge * EFFICIENCY
        else:
            usable = max((soc - MIN_SOC * cap) * EFFICIENCY, 0)
            discharge = min(d - g, pmax, usable)
            soc -= discharge / EFFICIENCY
            backup = d - g - discharge
        rows.append((soc / cap * 100, backup))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def hour_ranges(hours):
    out, start, prev = [], hours[0], hours[0]
    for h in list(hours[1:]) + [None]:
        if h is None or h != prev + 1:
            out.append(f"{start:02d}:00-{prev:02d}:59")
            start = h
        prev = h
    return ", ".join(out)


def battery_plan(day, gen_expected, factor=1.0):
    dem = demand_for(day)["expected"] * factor
    exp_soc, exp_backup = simulate(gen_expected, dem, factor)
    low_soc, low_backup = simulate(gen_expected * LOW_OUTPUT, dem, factor)
    short = np.where(low_backup > 0.1)[0]
    return {"demand_kwh": float(dem.sum()), "exp_backup": float(exp_backup.sum()), "low_backup": float(low_backup.sum()),
            "advice_hours": hour_ranges([int(h) for h in short]) if len(short) else None,
            "exp_soc": exp_soc, "low_soc": low_soc}


# ---------------- Charts ----------------
def chart_png(title, ylabel, expected, low=None, high=None, actual=None, extra=None, timestamps=None):
    n = len(expected)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(n)
    if low is not None and high is not None:
        ax.fill_between(x, low, high, alpha=0.25, color="#2e7d32", label="Forecast range")
    ax.plot(x, expected, color="#1b5e20", lw=2.2, label="Expected")
    if actual is not None:
        ax.plot(x, actual, color="#e65100", lw=1.8, ls="--", label="Actual")
    if extra:
        for label, series, style in extra:
            ax.plot(x, series, style, label=label)
    ax.set_title(title); ax.set_xlabel("Forecast hour"); ax.set_ylabel(ylabel)
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    if timestamps is not None:
        labels = [pd.Timestamp(timestamps[i]).strftime("%d %b %H:%M") for i in ticks]
    else:
        labels = [str(i) for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def friendly_area_name(name, lat=None, lon=None):
    """Return a short, familiar main-area label without changing GPS coordinates.

    The exact latitude/longitude remains the source of truth. This is only the
    human-facing label, so small localities such as Srinivasapura/Hemmigepura
    can be presented as the better-known Kengeri area.
    """
    raw = str(name or "").strip()
    text = raw.lower().replace("-", " ")

    # Familiar Bengaluru-area groupings. Text matching is preferred because
    # locality boundaries can overlap; coordinates are used as a fallback.
    groups = [
        ("Kengeri", ["kengeri", "hemmigepura", "srinivasapura", "srinivaspura",
                     "pattanagere", "patanagere", "kengeri satellite", "kengeri upanagar",
                     "kengeri upanagara", "kodipalya", "sunkalapalya", "konasandra"]),
        ("HSR Layout", ["hsr layout", "hsr sector", "hsr"]),
        ("BTM Layout", ["btm layout", "btm"]),
        ("Rajarajeshwari Nagar", ["rajarajeshwari nagar", "rr nagar", "rrnagar"]),
        ("Jayanagar", ["jayanagar"]),
        ("JP Nagar", ["jp nagar", "j p nagar"]),
        ("Banashankari", ["banashankari"]),
        ("Koramangala", ["koramangala"]),
        ("Indiranagar", ["indiranagar"]),
        ("Whitefield", ["whitefield"]),
        ("Electronic City", ["electronic city", "electroniccity"]),
        ("Yelahanka", ["yelahanka"]),
        ("Hebbal", ["hebbal"]),
        ("Marathahalli", ["marathahalli"]),
        ("Rajajinagar", ["rajajinagar"]),
        ("Malleshwaram", ["malleshwaram"]),
    ]
    for main, keys in groups:
        if any(k in text for k in keys):
            return f"{main}, Bengaluru"

    # Conservative coordinate fallbacks for the most requested Bengaluru
    # areas. These are display-only and do not alter lat/lon used for weather.
    try:
        la, lo = float(lat), float(lon)
        if 12.87 <= la <= 12.96 and 77.43 <= lo <= 77.54:
            return "Kengeri, Bengaluru"
        if 12.89 <= la <= 12.94 and 77.59 <= lo <= 77.66:
            if lo >= 77.615:
                return "HSR Layout, Bengaluru"
            return "BTM Layout, Bengaluru"
    except (TypeError, ValueError):
        pass

    return raw or "Current location"


def current_loc():
    """Chosen place plus simple plant-planning details from the generation form."""
    if request.args.get("reset") == "1":
        session.pop("loc", None)
    loc = dict(session.get("loc") or DEFAULT_LOC)
    if "user" in session:
        saved_settings = load_plant_settings(session["user"])
        if saved_settings:
            loc.update({k: v for k, v in saved_settings.items() if v is not None})
    try:
        if request.args.get("lat") and request.args.get("lon"):
            lat, lon = float(request.args["lat"]), float(request.args["lon"])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                loc.update({"lat": round(lat, 6), "lon": round(lon, 6), "name": (request.args.get("name") or "Chosen place")[:60],
                            "reference": abs(lat - REF_LAT) < 0.05 and abs(lon - REF_LON) < 0.05})
                if loc["reference"]:
                    loc["name"] = DEFAULT_LOC["name"]

        status = request.args.get("plant_status")
        if status in ("existing", "planning"):
            loc["plant_status"] = status
            if status == "planning":
                # A new user should not have to choose technical equipment details.
                loc["plant_type"] = "monocrystalline"
                loc["mounting"] = "fixed"
                loc["plant_type_known"] = False
                loc["mounting_known"] = False

        if request.args.get("purpose"):
            loc["purpose"] = request.args["purpose"][:40]
        if request.args.get("monthly_bill"):
            bill = float(request.args["monthly_bill"])
            if 0 <= bill <= 100000000:
                loc["monthly_bill"] = bill
        if request.args.get("area"):
            area = float(request.args["area"])
            if 0 < area <= 100000000:
                loc["available_area"] = area

        if loc.get("plant_status") == "planning":
            loc["kw"] = recommend_capacity(loc.get("monthly_bill", 0), loc.get("available_area"), loc.get("purpose", "home"))
            loc["capacity_source"] = "recommended"
        elif "kw" in request.args:
            raw_kw = request.args.get("kw", "").strip()
            if raw_kw:
                kw = float(raw_kw)
                if 0.1 <= kw <= 100000:
                    loc["kw"] = kw
                    loc["capacity_source"] = "user"
            else:
                # Existing-plant users may not know the nameplate size.
                # Use a clearly documented standard estimate instead of blocking.
                loc["kw"] = 5.0
                loc["capacity_source"] = "default"

        pt = request.args.get("plant_type")
        if pt == "unknown":
            loc["plant_type"] = "monocrystalline"
            loc["plant_type_known"] = False
        elif pt in PLANT_TYPES:
            loc["plant_type"] = pt
            loc["plant_type_known"] = True

        mt = request.args.get("mounting")
        if mt == "unknown":
            loc["mounting"] = "fixed"
            loc["mounting_known"] = False
        elif mt in MOUNTING_TYPES:
            loc["mounting"] = mt
            loc["mounting_known"] = True
    except (ValueError, TypeError):
        pass
    session["loc"] = loc
    if "user" in session and any(k in request.args for k in
        ("plant_status", "purpose", "monthly_bill", "monthly_units", "tariff", "area", "kw", "plant_type", "mounting")):
        persist_plant_settings(session["user"], loc)
    return loc


def parse_day(default):
    raw = request.args.get("date") or default.isoformat()
    try:
        return pd.Timestamp(raw).normalize(), raw
    except Exception:
        return None, raw


# ---------------- Pages ----------------
@app.route("/")
def index():
    # Always start the website at the login screen. This prevents a previously
    # restored browser session from silently opening the dashboard and skipping
    # the fresh-login GPS flow.
    session.clear()
    return redirect(url_for("login"))


def send_registration_otp(recipient, otp):
    """Send account-verification code through Brevo's HTTPS API.

    Render's free service may block outbound SMTP ports. Brevo's API uses
    HTTPS (port 443), so this avoids the SMTP connection timeout.
    """
    api_key = os.environ.get("BREVO_API_KEY", "").strip()
    sender = os.environ.get("SMTP_EMAIL", "").strip()
    sender_name = os.environ.get("SMTP_SENDER_NAME", "EcoGrid").strip() or "EcoGrid"

    if not api_key or not sender:
        raise RuntimeError(
            "Brevo email is not configured. Add BREVO_API_KEY (your Brevo API key) "
            "and SMTP_EMAIL (your verified sender email) to the Render environment."
        )

    payload = {
        "sender": {"name": sender_name, "email": sender},
        "to": [{"email": recipient}],
        "subject": "Your EcoGrid verification code",
        "textContent": (
            f"Your EcoGrid verification code is {otp}. It expires in 10 minutes. "
            "If you did not request this code, you can ignore this email."
        ),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=data,
        headers={
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            if response.status not in (200, 201, 202):
                raise RuntimeError(f"Brevo email API returned HTTP {response.status}.")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Brevo email API rejected the request (HTTP {exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Brevo over HTTPS: {exc.reason}") from exc


@app.route("/register", methods=["GET", "POST"])
def register():
    pending_email = session.get("registration_pending_email")
    if request.method == "POST":
        action = request.form.get("action", "send_otp")
        if action == "verify_otp":
            email = (request.form.get("email") or pending_email or "").strip().lower()
            otp = request.form.get("otp", "").strip()
            if not re.fullmatch(r"\d{6}", otp):
                flash("Enter the 6-digit verification code sent to your email.", "error")
            else:
                with db() as con:
                    pending = con.execute("SELECT * FROM pending_registrations WHERE lower(email)=?", (email,)).fetchone()
                    if not pending:
                        flash("Your verification request has expired. Please register again.", "error")
                        session.pop("registration_pending_email", None)
                    elif datetime.fromisoformat(pending["expires_at"]) < datetime.now(timezone.utc):
                        con.execute("DELETE FROM pending_registrations WHERE username=?", (pending["username"],))
                        flash("The OTP has expired. Please register again to receive a new code.", "error")
                        session.pop("registration_pending_email", None)
                    elif pending["attempts"] >= 5:
                        con.execute("DELETE FROM pending_registrations WHERE username=?", (pending["username"],))
                        flash("Too many incorrect attempts. Please register again.", "error")
                        session.pop("registration_pending_email", None)
                    elif not secrets.compare_digest(pending["otp_hash"], hashlib.sha256(otp.encode()).hexdigest()):
                        con.execute("UPDATE pending_registrations SET attempts=attempts+1 WHERE username=?", (pending["username"],))
                        flash("Incorrect OTP. Please check the code and try again.", "error")
                    else:
                        try:
                            con.execute("INSERT INTO users (username,email,password_hash) VALUES (?,?,?)",
                                        (pending["username"], pending["email"], pending["password_hash"]))
                            con.execute("DELETE FROM pending_registrations WHERE username=?", (pending["username"],))
                            session.pop("registration_pending_email", None)
                            flash("Email verified and account created successfully. You can now log in.", "ok")
                            return redirect(url_for("login"))
                        except sqlite3.IntegrityError:
                            flash("That username or email is already registered. Please log in or use different details.", "error")
        else:
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
                flash("Username must be 3–30 characters and use only letters, numbers, dots, underscores, or hyphens.", "error")
            elif not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
                flash("Enter a valid email address.", "error")
            elif len(password) < 6:
                flash("Use a password with at least 6 characters.", "error")
            else:
                with db() as con:
                    exists = con.execute("SELECT 1 FROM users WHERE lower(username)=? OR lower(email)=?", (username.lower(), email)).fetchone()
                if exists:
                    flash("That username or email is already registered. Please log in or use different details.", "error")
                else:
                    otp = f"{secrets.randbelow(1000000):06d}"
                    try:
                        send_registration_otp(email, otp)
                        now = datetime.now(timezone.utc)
                        with db() as con:
                            con.execute("INSERT OR REPLACE INTO pending_registrations (username,email,password_hash,otp_hash,expires_at,attempts,created_at) VALUES (?,?,?,?,?,?,?)",
                                        (username, email, generate_password_hash(password), hashlib.sha256(otp.encode()).hexdigest(),
                                         (now + timedelta(minutes=10)).isoformat(), 0, now.isoformat()))
                        session["registration_pending_email"] = email
                        flash("A 6-digit OTP has been sent to your email. Enter it below to verify your account.", "ok")
                        pending_email = email
                    except Exception as exc:
                        app.logger.exception("Unable to send EcoGrid registration OTP")
                        flash(str(exc) if isinstance(exc, RuntimeError) else "Could not send the verification email. Check SMTP settings and try again.", "error")
    return render_template("register.html", pending_email=pending_email)


@app.route("/login", methods=["GET", "POST"])
def login():
    # Visiting the login page explicitly must never leave an old authenticated
    # browser session active. A successful POST below creates a new session.
    if request.method == "GET":
        session.clear()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with db() as con:
            row = con.execute("SELECT * FROM users WHERE lower(username) = ?", (username.lower(),)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            session.clear()
            session["user"] = row["username"]
            with db() as con:
                saved = con.execute("SELECT name, lat, lon FROM user_locations WHERE username = ?", (row["username"],)).fetchone()
            if saved:
                session["loc"] = {**DEFAULT_LOC, "name": saved["name"], "lat": float(saved["lat"]), "lon": float(saved["lon"]), "reference": False}
            return redirect(url_for("home", fresh_location=1))
        flash("Incorrect username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    # Immediately after login, the first Home render is a fresh-phone-location
    # gate. Do not render or use yesterday's saved/search location during this
    # first pass. The browser obtains the current device coordinates, saves them,
    # and then reloads /home normally.
    fresh_location = request.args.get("fresh_location") == "1"
    saved = None
    if not fresh_location:
        with db() as con:
            row = con.execute("SELECT name, lat, lon, updated_at FROM user_locations WHERE username = ?", (session["user"],)).fetchone()
        if row:
            saved = {"name": row["name"], "lat": float(row["lat"]), "lon": float(row["lon"]), "updated_at": row["updated_at"]}
    weather = None
    forecast_energy = None
    daily_consumption = None
    balance = None
    plant_capacity = 5.0
    if "user" in session:
        saved_plant = load_plant_settings(session["user"])
        if saved_plant:
            plant_capacity = float(saved_plant.get("kw") or 5.0)
    if saved:
        try:
            wf, offset, current = fetch_weather_forecast(saved["lat"], saved["lon"])
            cur_temp = float(current.get("temperature_2m", wf.iloc[0]["temperature"]))
            daily = current.get("daily", {}) or {}
            dmax = float(daily.get("temperature_max", [cur_temp])[0] or cur_temp)
            dmin = float(daily.get("temperature_min", [cur_temp])[0] or cur_temp)
            code = int(current.get("weather_code", 0) or 0)
            desc_map = {
                0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
                61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
                80: "Rain showers", 81: "Rain showers", 82: "Heavy rain showers", 95: "Thunderstorm",
                96: "Thunderstorm with hail", 99: "Thunderstorm with hail"
            }
            hourly_times = current.get("hourly_times", [])
            # Find the first hourly point at or after the current local hour.
            now_iso = str(current.get("time", ""))
            start_i = 0
            if now_iso and hourly_times:
                for i, t in enumerate(hourly_times):
                    if t >= now_iso:
                        start_i = i
                        break
            hourly = []
            for i in range(start_i, min(start_i + 6, len(hourly_times))):
                ts = pd.Timestamp(hourly_times[i])
                hourly.append({
                    "time": ts.strftime("%I %p").lstrip("0"),
                    "temperature": round(float(current["hourly_temperature"][i])),
                    "cloud": round(float(current["hourly_cloud"][i])),
                    "rain": round(float(current["hourly_precipitation"][i]), 1),
                    "code": int(current.get("hourly_weather_code", [0] * len(hourly_times))[i]) if current.get("hourly_weather_code") else 0
                })
            daily_rows = []
            for i, day in enumerate(daily.get("time", [])[:4]):
                daily_rows.append({
                    "date": pd.Timestamp(day).strftime("%a"),
                    "date_num": pd.Timestamp(day).strftime("%d %b"),
                    "max": round(float(daily["temperature_max"][i])),
                    "min": round(float(daily["temperature_min"][i])),
                    "description": desc_map.get(int(daily["weather_code"][i]), "Weather"),
                    "rain": round(float(daily["precipitation_sum"][i]), 1)
                })
            weather = {
                "temperature": round(cur_temp),
                "feels": round(float(current.get("apparent_temperature", cur_temp))),
                "humidity": round(float(current.get("relative_humidity_2m", 0))),
                "cloud": round(float(current.get("cloud_cover", 0))),
                "wind": round(float(current.get("wind_speed_10m", 0)), 1),
                "rain": round(float(current.get("precipitation", 0)), 1),
                "solar": round(float(current.get("shortwave_radiation", 0))),
                "condition": desc_map.get(code, "Weather"),
                "icon": "☀" if code == 0 else ("🌧" if code >= 51 else ("☁" if code >= 3 else "⛅")),
                "max": round(dmax), "min": round(dmin),
                "hourly": hourly, "daily": daily_rows,
                "timezone": current.get("timezone", ""),
                "source": current.get("weather_source", "Open-Meteo Best Match"),
                "lat": current.get("requested_latitude", saved["lat"]),
                "lon": current.get("requested_longitude", saved["lon"]),
                "resolved_lat": current.get("resolved_latitude", saved["lat"]),
                "resolved_lon": current.get("resolved_longitude", saved["lon"]),
                "updated": current.get("weather_generated_at", "")
            }
        except Exception as e:
            weather = None
        try:
            loc = dict(session.get("loc") or DEFAULT_LOC)
            loc.update({"lat": saved["lat"], "lon": saved["lon"], "name": saved["name"]})
            g = generation_for(pd.Timestamp.now().normalize(), loc, horizon=24)
            forecast_energy = float(g["energy"])
            plant_capacity = float(g["capacity"])
        except Exception:
            pass
        try:
            # Use the user's saved bill-based daily estimate when available.
            loc = session.get("loc") or {}
            bill = loc.get("monthly_bill")
            if bill is not None:
                from_bill = bill_based_consumption(float(bill))
                daily_consumption = float(from_bill["daily_kwh"]) if from_bill else None
        except Exception:
            pass
        if forecast_energy is not None and daily_consumption is not None:
            balance = forecast_energy - daily_consumption
    return render_template("home.html", saved_location=saved, weather=weather, forecast_energy=forecast_energy, daily_consumption=daily_consumption, balance=balance, plant_capacity=plant_capacity)

@app.route("/location")
@login_required
def location():
    saved = None
    with db() as con:
        row = con.execute("SELECT name, lat, lon, updated_at FROM user_locations WHERE username = ?", (session["user"],)).fetchone()
    if row:
        saved = {"name": row["name"], "lat": float(row["lat"]), "lon": float(row["lon"]), "updated_at": row["updated_at"]}
    return render_template("location.html", saved_location=saved)

@app.route("/save-location", methods=["POST"])
@login_required
def save_location():
    try:
        name = (request.form.get("name") or "Chosen location").strip()[:100]
        lat = float(request.form.get("lat", "")); lon = float(request.form.get("lon", ""))
        name = friendly_area_name(name, lat, lon)[:100]
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError
        now = datetime.now(timezone.utc).isoformat()
        with db() as con:
            con.execute("INSERT INTO user_locations(username,name,lat,lon,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(username) DO UPDATE SET name=excluded.name,lat=excluded.lat,lon=excluded.lon,updated_at=excluded.updated_at", (session["user"], name or "Chosen location", lat, lon, now))
        session["loc"] = {**DEFAULT_LOC, "name": name or "Chosen location", "lat": lat, "lon": lon, "reference": False}
        flash(f"Saved location: {name or 'Chosen location'}", "ok")
        if request.form.get("next") == "location":
            return redirect(url_for("location"))
        return redirect(url_for("home"))
    except (ValueError, TypeError):
        flash("Please choose a valid location.", "error")
        return redirect(url_for("home"))


@app.route("/generation")
@login_required
def generation():
    loc = current_loc()
    try:
        horizon = 48 if int(request.args.get("horizon", "24")) == 48 else 24
    except ValueError:
        horizon = 24
    ctx = {
        "date": datetime.now().date().isoformat(), "error": None, "loc": loc,
        "coverage": cloud_live.coverage(loc["lat"], loc["lon"]),
        "horizon": horizon, "plant_types": PLANT_TYPES, "mounting_types": MOUNTING_TYPES,
    }
    try:
        g = generation_for(pd.Timestamp.now().normalize(), loc, horizon=horizon)
    except ValueError as e:
        ctx["error"] = str(e)
        return render_template("generation.html", **ctx)
    session["loc"] = loc
    persist_plant_settings(session["user"], loc)
    # Bill-based energy outlook: use the user's actual monthly units when supplied,
    # otherwise estimate units from the monthly bill and an editable approximate tariff.
    bill_data = bill_based_consumption(loc.get("monthly_bill"), loc.get("monthly_units"), loc.get("tariff", 8.0))
    representative_demand = None
    energy_balance = None
    if bill_data:
        representative_demand = float(bill_data["daily_kwh"] * horizon / 24.0)
        energy_balance = float(g["energy"] - representative_demand)
    save_forecast(session["user"], "solar_generation", loc, datetime.now().date().isoformat(),
                  horizon, g["energy"], consumption=representative_demand, capacity=g["capacity"],
                  details={"low_kwh": g["low_energy"], "high_kwh": g["high_energy"],
                           "peak_kw": g["peak"], "weather": g["situation"]})
    ctx.update({
        "mode": g["mode"], "accuracy": None, "band_note": g["band_note"],
        "chart": chart_png(f"Solar forecast - {loc['name']}", "Power (kW)", g["expected"], g["low"], g["high"],
                            timestamps=g["timestamps"]),
        "total_exp": g["energy"], "total_low": g["low_energy"], "total_high": g["high_energy"],
        "peak": g["peak"], "peak_hour": g["peak_hour"], "total_actual": None, "day_mae": None,
        "representative_demand": representative_demand,
        "energy_balance": energy_balance,
        "bill_data": bill_data,
        "is_live": True, "weather_situation": g["situation"], "clear_hours": g["clear_hours"],
        "cloudy_hours": g["cloudy_hours"], "rainy_hours": g["rainy_hours"],
        "avg_cloud": g["avg_cloud"], "avg_temp": g["avg_temp"],
        "plant_type_label": g["plant_type_label"], "mounting_label": g["mounting_label"],
        "capacity": g["capacity"], "forecast_hours": horizon,
    })
    return render_template("generation.html", **ctx)


@app.route("/api/reverse-geocode")
@login_required
def api_reverse_geocode():
    """Return the most detailed available label for the exact browser coordinates.

    Reverse-geocoders can disagree about locality boundaries. Coordinates
    remain the source of truth; the returned name is only a human-readable map
    label. The UI may show a broader familiar area together with the mapped
    locality so the name does not hide the exact GPS position.
    """
    try:
        lat = float(request.args.get("lat", ""))
        lon = float(request.args.get("lon", ""))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Coordinates are outside the valid range.")

        # OSM reverse lookup at street/building detail (zoom 18 instead of 14).
        params = urllib.parse.urlencode({
            "lat": f"{lat:.7f}", "lon": f"{lon:.7f}", "format": "jsonv2",
            "zoom": 18, "addressdetails": 1, "accept-language": "en"
        })
        osm_data = {}
        try:
            req = urllib.request.Request(
                "https://nominatim.openstreetmap.org/reverse?" + params,
                headers={"User-Agent": "EcoGridAI/1.1 (solar forecast location lookup)", "Accept-Language": "en"})
            with urllib.request.urlopen(req, timeout=8) as response:
                osm_data = json.load(response)
        except Exception:
            pass

        addr = osm_data.get("address", {}) if isinstance(osm_data, dict) else {}
        # Keep the nearest street/neighbourhood first. The previous code used
        # zoom 14 and could return a broad/adjacent suburb for a precise point.
        street = " ".join(x for x in [addr.get("house_number"), addr.get("road"),
                                       addr.get("pedestrian"), addr.get("residential")] if x)
        locality = (addr.get("neighbourhood") or addr.get("quarter") or addr.get("suburb") or
                    addr.get("hamlet") or addr.get("village") or addr.get("residential") or
                    addr.get("town") or addr.get("city_district") or addr.get("city") or
                    addr.get("municipality") or addr.get("county") or "")

        city = (addr.get("city") or addr.get("town") or addr.get("village") or
                addr.get("municipality") or addr.get("city_district") or "")
        state = addr.get("state") or addr.get("region") or ""
        country = addr.get("country") or ""
        parts = [street, locality, city, state, country]
        label = ", ".join(dict.fromkeys(x.strip() for x in parts if x and x.strip()))
        provider = "OpenStreetMap" if label else ""

        # Fallback or supplementary locality lookup; this provider often returns
        # locality/subLocality separately from the administrative district.
        try:
            bc_params = urllib.parse.urlencode({"latitude": f"{lat:.7f}", "longitude": f"{lon:.7f}", "localityLanguage": "en"})
            bc_req = urllib.request.Request(
                "https://api.bigdatacloud.net/data/reverse-geocode-client?" + bc_params,
                headers={"User-Agent": "EcoGridAI/1.1"})
            with urllib.request.urlopen(bc_req, timeout=6) as response:
                bc = json.load(response)
            bc_locality = bc.get("locality") or bc.get("localityInfo", {}).get("informative", [{}])[0].get("name", "")
            bc_city = bc.get("city") or bc.get("locality") or ""
            bc_state = bc.get("principalSubdivision") or ""
            bc_country = bc.get("countryName") or ""
            # Prefer the more local BDC locality for the short UI label when it
            # is available; keep coordinates as the authoritative location.
            if bc_locality and not str(bc_locality).strip().lower() in {"bengaluru", "bangalore"}:
                locality = bc_locality
                city = bc_city or city
                label = ", ".join(dict.fromkeys(x.strip() for x in [street, locality, city, state or bc_state, country or bc_country] if x and x.strip()))
                provider = "OpenStreetMap + BigDataCloud"
            if not label:
                label = ", ".join(x for x in [bc_locality, bc_city, bc_state, bc_country] if x)
                provider = "BigDataCloud"
            elif not locality and bc_locality:
                # Only supplement a missing locality; don't overwrite detailed street data.
                label = ", ".join(dict.fromkeys(x for x in [street, bc_locality, bc_city, state or bc_state, country or bc_country] if x))
                provider = "OpenStreetMap + BigDataCloud"
        except Exception:
            pass

        if not label:
            label = osm_data.get("display_name") or "Address unavailable; use GPS coordinates"
        friendly = friendly_area_name(label, lat, lon)
        return jsonify({"ok": True, "name": friendly, "display_name": label,
                        "address": addr, "provider": provider,
                        "latitude": lat, "longitude": lon,
                        "note": "Area names depend on map data. Verify the pin and GPS accuracy; a browser on a desktop may estimate location from Wi-Fi or IP rather than a GPS chip."})
    except Exception:
        return jsonify({"ok": False, "message": "GPS coordinates were obtained, but address lookup failed. Use the coordinates and Google Maps pin to verify the location."})


@app.route("/api/geocode")
@login_required
def api_geocode():
    q = (request.args.get("q") or "").strip()
    if len(q) < 3:
        return jsonify({"ok": False, "message": "Type at least 3 letters."})
    try:
        places = search_places(q)
    except Exception as e:
        return jsonify({"ok": False, "message": "Could not search places: " + str(e)[:100]})
    return jsonify({"ok": True, "places": places} if places else {"ok": False, "message": "No place found. Try another spelling."})


@app.route("/consumption")
@login_required
def consumption():
    """Show the ML demand forecast plus a user-adjustable appliance energy plan."""
    tomorrow = local_today() + timedelta(days=1)
    day, raw = parse_day(request.args.get("date") or tomorrow)
    appliances = [
        {"key": "fridge", "name": "Refrigerator", "watts": 100, "essential": True, "default": 24,
         "description": "Essential — average running power estimate"},
        {"key": "lights", "name": "LED lights (group)", "watts": 50, "essential": True, "default": 5,
         "description": "Essential — 5 lights at about 10 W each"},
        {"key": "fan", "name": "Fan", "watts": 75, "essential": True, "default": 8,
         "description": "Essential for comfort"},
        {"key": "tv", "name": "TV / entertainment", "watts": 100, "essential": False, "default": 3,
         "description": "Can be reduced or used later"},
        {"key": "washer", "name": "Washing machine", "watts": 500, "essential": False, "default": 1,
         "description": "Run a full load during sunny hours"},
        {"key": "ac", "name": "Air conditioner", "watts": 1500, "essential": False, "default": 4,
         "description": "Reduce slightly if safe and comfortable"},
        {"key": "heater", "name": "Water heater / geyser", "watts": 2000, "essential": False, "default": 1,
         "description": "Heat water during solar peak hours"},
        {"key": "ev", "name": "EV charging", "watts": 3000, "essential": False, "default": 0,
         "description": "Delay charging to solar-rich hours"},
        {"key": "other", "name": "Other appliance (custom)", "watts": 0, "essential": False, "default": 0,
         "description": "Enter its rated power and hours below. Consider whether this appliance is essential or flexible."},
    ]
    with db() as con:
        saved_profile = con.execute("SELECT * FROM consumption_profiles WHERE username=?", (session["user"],)).fetchone()
    saved_profile = dict(saved_profile) if saved_profile else {}
    try:
        saved_plan = json.loads(saved_profile.get("appliance_plan_json") or "{}")
    except (TypeError, ValueError):
        saved_plan = {}
    for item in appliances:
        previous = saved_plan.get(item["key"], {})
        if previous:
            item["default"] = previous.get("hours", item["default"])
            if item["key"] == "other":
                item["watts"] = previous.get("watts", 0)
                item["name"] = previous.get("name", item["name"])
    ctx = {"date": raw, "error": None, "appliances": appliances,
           "system_kw": request.args.get("system_kw", saved_profile.get("system_kw", "5")), "plan_ready": False,
           "generation_error": None, "generation_kwh": None,
           "monthly_bill": request.args.get("monthly_bill", saved_profile.get("monthly_bill", session.get("energy_monthly_bill", ""))),
           "monthly_units": request.args.get("monthly_units", saved_profile.get("monthly_units", session.get("energy_monthly_units", ""))),
           "tariff": request.args.get("tariff", saved_profile.get("tariff", session.get("energy_tariff", "8")))}
    if day is None:
        ctx["error"] = "That is not a valid date."
        return render_template("consumption.html", **ctx)

    # Primary consumption estimate comes from the electricity bill / monthly kWh.
    bill_data = bill_based_consumption(ctx["monthly_bill"], ctx["monthly_units"], ctx["tariff"])
    ctx["bill_data"] = bill_data
    if bill_data:
        session["energy_monthly_bill"] = ctx["monthly_bill"]
        session["energy_monthly_units"] = ctx["monthly_units"]
        session["energy_tariff"] = ctx["tariff"]

    # Keep the historical AI profile as a secondary reference, but do not use it
    # as the user's personal consumption estimate.
    d = demand_for(day)
    ctx.update({
        "mode": d["mode"], "mape": DEM["test_mape"],
        "chart": chart_png(f"Energy consumption forecast - {day:%d %b %Y}", "Demand (kW, scaled to plant size)",
                           d["expected"], d["low"], d["high"], d["actual"]),
        "total_exp": float(d["expected"].sum()), "peak": float(d["expected"].max()),
        "peak_hour": int(d["expected"].argmax()),
        "total_actual": None if d["actual"] is None else float(d["actual"].sum())})

    # Appliance energy = rated power (kW) x user-entered operating hours.
    custom_total = 0.0
    for item in appliances:
        try:
            hours = float(request.args.get(item["key"] + "_hours", item["default"]))
        except (TypeError, ValueError):
            hours = float(item["default"])
        hours = min(24.0, max(0.0, hours))
        if item["key"] == "other":
            item["custom_name"] = (request.args.get("other_name") or "Other appliance").strip()[:60] or "Other appliance"
            try:
                item["watts"] = min(1000000.0, max(0.0, float(request.args.get("other_watts", "0"))))
            except (TypeError, ValueError):
                item["watts"] = 0.0
            item["name"] = item["custom_name"]
            item["description"] = "Custom appliance entered by you"
        item["hours"] = hours
        item["energy"] = item["watts"] / 1000.0 * hours
        custom_total += item["energy"]
    ctx["appliance_total"] = custom_total
    ctx["plan_ready"] = True
    plan = {a["key"]: {"name": a["name"], "watts": a["watts"], "hours": a["hours"],
                       "essential": a["essential"], "energy_kwh": a["energy"]} for a in appliances}
    with db() as con:
        con.execute("""INSERT INTO consumption_profiles
            (username,monthly_bill,monthly_units,tariff,system_kw,appliance_plan_json,updated_at)
            VALUES(?,?,?,?,?,?,?) ON CONFLICT(username) DO UPDATE SET
            monthly_bill=excluded.monthly_bill,monthly_units=excluded.monthly_units,
            tariff=excluded.tariff,system_kw=excluded.system_kw,
            appliance_plan_json=excluded.appliance_plan_json,updated_at=excluded.updated_at""",
            (session["user"], ctx["monthly_bill"] or None, ctx["monthly_units"] or None,
             float(ctx["tariff"] or 8), float(ctx["system_kw"] or 5), json.dumps(plan),
             datetime.now(timezone.utc).isoformat()))

    try:
        system_kw = float(request.args.get("system_kw", "5"))
        if not (0.1 <= system_kw <= 100000):
            raise ValueError("Enter a solar system size between 0.1 and 100000 kW.")
        loc = dict(current_loc())
        loc["kw"] = system_kw
        solar = generation_for(day, loc)
        generation_kwh = float(np.sum(solar["expected"]))
        ctx["generation_kwh"] = generation_kwh
        ctx["system_kw"] = system_kw
        ctx["energy_balance"] = generation_kwh - custom_total
        ctx["shortage_kwh"] = max(0.0, custom_total - generation_kwh)

        # Reduce flexible loads first. Essential appliances are never recommended to stop.
        reducible_order = ["ev", "heater", "washer", "tv", "ac"]
        remaining = ctx["shortage_kwh"]
        suggestions = []
        by_key = {a["key"]: a for a in appliances}
        for key in reducible_order:
            item = by_key[key]
            if remaining <= 0.01 or item["hours"] <= 0:
                continue
            kw = item["watts"] / 1000.0
            reduce_hours = min(item["hours"], remaining / kw)
            saved = reduce_hours * kw
            if saved > 0.01:
                suggestions.append({"name": item["name"], "hours": reduce_hours, "saved": saved,
                                    "description": item["description"]})
                remaining -= saved
        ctx["suggestions"] = suggestions
        ctx["uncovered_shortage"] = max(0.0, remaining)
        ctx["generation_mode"] = solar["mode"]
        save_forecast(session["user"], "consumption_plan", current_loc(), day.date().isoformat(),
                      24, generation_kwh, consumption=custom_total, capacity=system_kw,
                      details={"appliances": plan, "shortage_kwh": ctx["shortage_kwh"],
                               "uncovered_shortage_kwh": ctx["uncovered_shortage"]})
    except Exception as e:
        ctx["generation_error"] = str(e)[:220]
        ctx["suggestions"] = []
        ctx["uncovered_shortage"] = None
    return render_template("consumption.html", **ctx)

@app.route("/api/profile")
@login_required
def api_profile():
    """Return the signed-in user's saved EcoGrid settings and location."""
    with db() as con:
        settings = con.execute("SELECT * FROM user_settings WHERE username=?", (session["user"],)).fetchone()
        location = con.execute("SELECT name,lat,lon,updated_at FROM user_locations WHERE username=?", (session["user"],)).fetchone()
        consumption = con.execute("SELECT monthly_bill,monthly_units,tariff,system_kw,appliance_plan_json,updated_at FROM consumption_profiles WHERE username=?", (session["user"],)).fetchone()
    return jsonify({"ok": True, "username": session["user"],
                    "plant": dict(settings) if settings else None,
                    "location": dict(location) if location else None,
                    "consumption": dict(consumption) if consumption else None})


@app.route("/api/history")
@login_required
def api_history():
    """Return recent saved forecast records for the signed-in user only."""
    try:
        limit = min(100, max(1, int(request.args.get("limit", "20"))))
    except ValueError:
        limit = 20
    with db() as con:
        rows = con.execute("""SELECT id,forecast_type,location_name,latitude,longitude,forecast_date,
            horizon_hours,generation_kwh,consumption_kwh,energy_balance_kwh,capacity_kw,created_at
            FROM forecast_history WHERE username=? ORDER BY id DESC LIMIT ?""",
            (session["user"], limit)).fetchall()
    return jsonify({"ok": True, "records": [dict(r) for r in rows]})


@app.route("/api/cloud")
@login_required
def api_cloud():
    try:
        loc = session.get("loc") or DEFAULT_LOC
        return jsonify(cloud_live.get_outlook(lat=loc["lat"], lon=loc["lon"], offset_seconds=loc.get("offset"),
                                              reference=loc["reference"], demo=request.args.get("demo") == "1"))
    except Exception as e:
        return jsonify({"ok": False, "message": "Cloud outlook failed: " + str(e)[:150]})


if __name__ == "__main__":
    app.run(debug=True)
