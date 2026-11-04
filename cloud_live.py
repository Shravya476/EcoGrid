"""Live cloud outlook from the Himawari-9 weather satellite (infrared Band 13), for any location it can see.
  - Reference plant (Alice Springs), 08:00-15:59 local time: cloud pictures + sunshine forecast for the next 3 hours.
  - Any other place or hour: cloud pictures + a simple 'clouds increasing / clearing / steady' reading
    (the sunshine model was trained at the reference plant only, and infrared shows clouds day and night).
Everything here is optional - if a package, model or the Internet is missing, the app shows a message instead."""
import os, io, bz2, math, shutil, base64, tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
HALF = 0.64                                              # half-width of the picture in degrees (about 70 km)
BUCKET = "noaa-himawari9"
SAT_LON = 140.7                                          # longitude the satellite sits above (on the equator)
REF_LAT, REF_LON = -23.76, 133.87
REF_OFFSET = timedelta(hours=9, minutes=30)              # Alice Springs time


def view_angle(lat, lon):
    """Angle (degrees) between the point and the spot directly under the satellite. Bigger = closer to the edge of its view."""
    la, lo = math.radians(lat), math.radians(lon - SAT_LON)
    return math.degrees(math.acos(max(-1, min(1, math.cos(la) * math.cos(lo)))))


def coverage(lat, lon):
    """('ok' | 'edge' | 'no', message)."""
    a = view_angle(lat, lon)
    if a > 70:
        return "no", ("This place is outside the view of the Himawari-9 satellite (it covers Australia, East Asia and the "
                      "western Pacific). Another satellite would be needed, which this app does not use yet.")
    if a > 60:
        return "edge", "This place is near the edge of the satellite's view, so the pictures are stretched and rough."
    return "ok", ""


def strips_for(lat, lon):
    """The satellite stores the Earth in 10 horizontal strips. Find the strip(s) covering the picture around the point."""
    def row(la):
        phi_deg = math.radians(la)
        req, rpol, H = 6378.137, 6356.752, 42164.0
        lo = math.radians(lon - SAT_LON)
        phi = math.atan((rpol ** 2 / req ** 2) * math.tan(phi_deg))
        rc = rpol / math.sqrt(1 - ((req ** 2 - rpol ** 2) / req ** 2) * math.cos(phi) ** 2)
        sx = H - rc * math.cos(phi) * math.cos(lo)
        sy = -rc * math.cos(phi) * math.sin(lo)
        sz = rc * math.sin(phi)
        y = math.asin(sz / math.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
        return 2750 - y / 56e-6                               # 5500 rows at 2 km, row 2750 = equator
    r1, r2 = sorted([row(lat - HALF), row(lat + HALF)])
    first, last = int(r1 // 550) + 1, int(r2 // 550) + 1
    return [s for s in range(max(first, 1), min(last, 10) + 1)]


def _key_for(ts, seg):
    name = f"HS_H09_{ts:%Y%m%d_%H%M}_B13_FLDK_R20_S{seg:02d}10.DAT"
    return f"AHI-L1b-FLDK/{ts:%Y/%m/%d/%H%M}/{name}.bz2", name


def last_daytime_slot(t):
    """Latest satellite slot at or before t (UTC) that is exactly 15:00 Alice Springs time."""
    cand = t
    for _ in range(300):
        local = cand + REF_OFFSET
        if local.hour == 15 and local.minute == 0:
            return cand
        cand -= timedelta(minutes=10)
    return None


def relative_cloudiness(img):
    """Share of the picture that is clearly colder than its own warm ground (works day and night)."""
    warm = np.nanpercentile(img, 90)
    return float(np.nanmean(img < warm - 8))


def make_picture(frames, times_local):
    """Three infrared pictures, oldest to newest. Clouds are cold, so they look BRIGHT. Red mark = the chosen place."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = np.nanpercentile(frames, 1), np.nanpercentile(frames, 99)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    for ax, img, tl in zip(axes, frames, times_local):
        ax.imshow(img, cmap="gray_r", vmin=lo, vmax=hi)
        ax.plot(32, 32, marker="+", color="red", markersize=14, mew=2)
        ax.set_title(tl.strftime("%H:%M local"), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def analyse(frames, t, offset, age, demo, reference, models=None, now_model=None, edge_note=""):
    """frames: 3 images (t-2h, t-1h, t). Returns the dictionary the web page shows."""
    local = t + offset
    times_local = [t - timedelta(hours=k) + offset for k in (2, 1, 0)]
    out = {"ok": True, "demo": demo, "newest_local": local.strftime("%Y-%m-%d %H:%M"), "minutes_old": age,
           "tz_label": "Alice Springs time" if reference else "local time at this place",
           "image": make_picture(frames, times_local)}
    cloud_share = [relative_cloudiness(f) for f in frames]
    out["cloud_share"] = [round(c * 100) for c in cloud_share]

    if reference and 8 <= local.hour <= 15 and models is not None and now_model is not None:   # sunshine forecast
        import cloud_features
        est_now = float(np.clip(now_model.predict(
            cloud_features.estimate_features(frames[-1], local.hour, local.month).reshape(1, -1))[0], 0, 1.2))
        row = cloud_features.example_features(frames, est_now, local.hour, local.month).reshape(1, -1)
        pred = [float(np.clip(est_now + m.predict(row)[0], 0, 1.2)) for m in models]
        drop, rise = est_now - min(pred), max(pred) - est_now
        top = min(max(pred), 1.0) * 100
        if drop > 0.20:
            alert, level = "Cloud approaching: sunshine is expected to fall from about %.0f%% to %.0f%% within 3 hours." % (est_now * 100, min(pred) * 100), "warn"
        elif drop > 0.08:
            alert, level = ("Some cloud may drift in: sunshine may ease from about %.0f%% to %.0f%% within 3 hours "
                            "(a change this size is within the model's usual error)." % (est_now * 100, min(pred) * 100)), "info"
        elif rise > 0.20:
            alert, level = "Skies are clearing: sunshine is expected to rise from about %.0f%% to %.0f%%." % (est_now * 100, top), "good"
        elif rise > 0.08:
            alert, level = "Skies may clear a little: sunshine may rise from about %.0f%% to %.0f%%." % (est_now * 100, top), "info"
        else:
            alert, level = "No big cloud change expected in the next 3 hours.", "good"
        if demo:
            alert = "DEMO (not live, last daytime images): " + alert
        out.update({"mode": "forecast", "now": round(est_now, 2), "forecast": [round(p, 2) for p in pred],
                    "alert": alert, "level": level})
        return out

    change = cloud_share[-1] - cloud_share[0]                                          # cloud trace only
    if change > 0.10:
        alert, level = "Clouds are increasing over the area (cloudy share rose from %.0f%% to %.0f%% in 2 hours)." % (cloud_share[0] * 100, cloud_share[-1] * 100), "warn"
    elif change < -0.10:
        alert, level = "Clouds are clearing (cloudy share fell from %.0f%% to %.0f%% in 2 hours)." % (cloud_share[0] * 100, cloud_share[-1] * 100), "good"
    else:
        alert, level = "Cloud cover is steady (about %.0f%% of the area)." % (cloud_share[-1] * 100), "good"
    why = ("The sunshine forecast model was trained at the reference plant (Alice Springs) and works there from 08:00 to 15:59 local time. "
           "For other places and hours the app shows the cloud trace only." if not reference else
           "It is %s at the plant, outside the 08:00-15:59 window of the sunshine forecast, so only the cloud trace is shown." % local.strftime("%H:%M"))
    out.update({"mode": "trace", "alert": alert, "level": level,
                "note": (edge_note + " " if edge_note else "") + why + " The cloudy share is measured relative to the warm ground in the picture, "
                        "so it can miss a completely overcast sky, and it has not been tested outside the reference plant."})
    return out


def get_outlook(lat=REF_LAT, lon=REF_LON, offset_seconds=None, reference=True, demo=False):
    status, msg = coverage(lat, lon)
    if status == "no":
        return {"ok": False, "message": msg}
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError
        from satpy import Scene
        from pyresample import create_area_def
    except ImportError as e:
        return {"ok": False, "message": "The live cloud outlook needs extra packages (" + str(e) +
                "). Install them with: pip install satpy pyresample boto3"}

    offset = REF_OFFSET if reference else timedelta(seconds=(offset_seconds if offset_seconds is not None else round(lon / 15 * 3600)))
    models = now_model = None
    cloud_path, now_path = os.path.join(MODELS, "cloud_models.joblib"), os.path.join(MODELS, "cloud_now_model.joblib")
    if reference and os.path.exists(cloud_path) and os.path.exists(now_path):
        models, now_model = joblib.load(cloud_path)["models"], joblib.load(now_path)

    segs = strips_for(lat, lon)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
    area = create_area_def("place", {"proj": "longlat", "datum": "WGS84"},
                           area_extent=[lon - HALF, lat - HALF, lon + HALF, lat + HALF], shape=(64, 64), units="degrees")

    def exists(ts):
        try:
            s3.head_object(Bucket=BUCKET, Key=_key_for(ts, segs[0])[0])
            return True
        except ClientError:
            return False

    def fetch_crop(ts, tmp):
        paths = []
        for seg in segs:
            key, name = _key_for(ts, seg)
            bz, dat = os.path.join(tmp, name + ".bz2"), os.path.join(tmp, name)
            s3.download_file(BUCKET, key, bz)
            with bz2.open(bz) as fin, open(dat, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            paths.append(dat)
        scn = Scene(filenames=paths, reader="ahi_hsd")
        scn.load(["B13"])
        return scn.resample(area, resampler="nearest")["B13"].values.astype("float32")

    try:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        t = now - timedelta(minutes=now.minute % 10)
        for _ in range(9):
            if exists(t):
                break
            t -= timedelta(minutes=10)
        if demo and reference and not 8 <= (t + offset).hour <= 15:        # demo: use the last daytime images
            t = last_daytime_slot(t)
            if t is None or not exists(t):
                return {"ok": False, "message": "Could not find the last daytime images for the demo."}
        age = round((now - t).total_seconds() / 60)
        with tempfile.TemporaryDirectory() as tmp:
            frames = np.stack([fetch_crop(t - timedelta(hours=k), tmp) for k in (2, 1, 0)])
        return analyse(frames, t, offset, age, demo and reference, reference, models, now_model, msg if status == "edge" else "")
    except Exception as e:
        return {"ok": False, "message": "Could not get the satellite images: " + str(e)[:150]}
