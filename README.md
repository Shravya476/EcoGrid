# EcoGrid AI - Solar Generation & Future Energy Consumption

## What this version does

EcoGrid forecasts solar generation for a **new/existing solar plant even when the plant has no historical generation records**.

Flow:

1. User logs in.
2. User allows browser location or searches a place / enters coordinates.
3. User enters installed solar capacity in kW.
4. User selects solar panel type and mounting/tracking type.
5. EcoGrid requests the next **24 or 48 hours** of weather and solar-radiation forecasts from Open-Meteo.
6. The generation engine estimates hourly power from installed capacity, forecast solar radiation, temperature, module type and mounting type.
7. The dashboard shows expected energy, an uncertainty range, peak power and the expected climate/weather situation.
8. The existing optional Himawari-9 satellite cloud feature remains available as an additional cloud check.
9. The Energy Consumption page continues to provide appliance-based future energy planning and the representative demand model.

## Why no historical plant data is required

The old generation model was trained on one reference plant in Alice Springs and scaled its output to other locations. That was not appropriate for a new plant with no historical readings.

This version uses a **physics-informed, weather-driven estimate** instead. The plant's installed/nameplate capacity is the base, and the forecast radiation is used to estimate the fraction of rated power available during each hour. Temperature and plant characteristics are then used as adjustments.

This is an estimate, not a measured meter reading. Once the real plant starts collecting generation data, the project can later be recalibrated with local readings.

## Weather inputs

The forecast uses Open-Meteo hourly values including:

- Shortwave solar radiation (GHI)
- Direct normal irradiance (DNI)
- Diffuse radiation
- Cloud cover
- Temperature
- Relative humidity
- Wind speed
- Precipitation

Open-Meteo documents these solar-radiation and cloud variables in its forecast API documentation.

## Plant inputs

- Installed capacity (kW)
- Panel type: Monocrystalline, Polycrystalline or Thin-film
- Mounting: Fixed tilt, Single-axis tracking or Dual-axis tracking

The type/mounting adjustments are engineering assumptions, not measured characteristics of a specific manufacturer's module. The displayed uncertainty range is also an assumption around the weather-driven estimate.

## Live cloud feature

The existing Himawari-9 satellite cloud outlook has intentionally been kept. It is optional and can be unavailable outside its coverage area. The main 24/48-hour generation forecast does not depend on the satellite feature.

## Future energy consumption

The project keeps the appliance planner and representative ML demand forecast. The appliance planner is the better choice when the actual site's consumption history is unavailable because the user can enter expected appliance/load hours and compare them with forecast solar generation.

## Running

Install packages from `requirements.txt` and run:

```bash
python app.py
```

The app uses Flask, SQLite, Open-Meteo, pandas, numpy, scikit-learn, joblib and matplotlib.

For deployment, serve the Flask application using the platform's normal Python web-service command.

## Theme selection
EcoGrid now includes a Light / Dark theme switch in the top navigation. The selected theme is saved in the browser with localStorage and is restored automatically on the next visit.


## SQLite backend (expanded)

The app now uses SQLite as its persistent application database. By default the database
file is `users.db` beside `app.py`. Set `ECOGRID_DB_PATH` to an absolute path when the
hosting platform provides a persistent disk.

### Stored data

- `users`: unique usernames and Werkzeug password hashes (never plain-text passwords).
- `user_locations`: one saved GPS/search location per username, including coordinates.
- `user_settings`: plant capacity, panel type, mounting, plant status, purpose, bill,
  monthly units, tariff and available area.
- `consumption_profiles`: saved bill/unit inputs, system capacity and appliance plan.
- `forecast_history`: timestamped generation and consumption forecast summaries,
  including location, horizon, energy balance and relevant forecast details.
- `app_meta`: one-time database migration metadata.

The `/api/profile` endpoint returns the authenticated user's saved profile.
The `/api/history` endpoint returns that user's most recent forecast records
(20 by default; maximum 100). Both endpoints require login and only query the
currently authenticated username.

### Persistence on hosting platforms

A SQLite file stored on an ephemeral deployment filesystem can be lost when the
service is redeployed or restarted. Attach a persistent disk and configure
`ECOGRID_DB_PATH` to a file on that disk (for example `/var/data/ecogrid.db`).
Do not place the database inside a temporary/cache directory. Back up the SQLite
file periodically. The existing HTML templates, CSS, page routes and visual
components have not been redesigned for this backend update.

### Run locally

```bash
pip install -r requirements.txt
python app.py
```

The first run creates or migrates the SQLite tables automatically.


## Email OTP registration

Registration now asks for username, email and password. EcoGrid sends a six-digit OTP; the user account is inserted into SQLite only after the OTP is verified. OTPs expire after 10 minutes and are limited to five attempts. Existing login remains username + password. Existing user rows are preserved; their email column is empty until they register with a verified email in a future account flow.

Configure SMTP as environment variables before running the app (do not commit real credentials). A safe placeholder file, `.env.example`, is included; Flask does not automatically load it, so set the variables in your terminal or hosting settings:

- `SMTP_HOST` — SMTP server hostname
- `SMTP_PORT` — usually `587` for STARTTLS or `465` for SSL
- `SMTP_EMAIL` — sender email address
- `SMTP_PASSWORD` — sender mailbox/app password
- `SMTP_USE_TLS=true` — default for port 587; set `false` when using SSL
- `SMTP_USE_SSL=true` — use for port 465 (and set `SMTP_USE_TLS=false`)

For Gmail, enable 2-Step Verification and create a Google App Password; use `smtp.gmail.com`, port `587`, TLS enabled. Do not use your normal Google account password.

Windows CMD example (replace the values locally):

```bat
set SMTP_HOST=smtp.gmail.com
set SMTP_PORT=587
set SMTP_EMAIL=youraddress@gmail.com
set SMTP_PASSWORD=your-16-character-app-password
set SMTP_USE_TLS=true
set SMTP_USE_SSL=false
python app.py
```

These `set` values apply only to that CMD window. For deployment, add the same variables in your hosting provider's environment/secrets settings.


## Brevo email OTP setup (one time)

1. Create/sign in to a Brevo account.
2. In Brevo, open **Settings -> Senders, Domains & IPs** and verify the sender address you want EcoGrid to send from. For production, authenticate your sending domain (SPF/DKIM).
3. Open **Settings -> SMTP & API -> SMTP**. Copy the SMTP login and generate an **SMTP key**. Do not use an API key or your Brevo account password.
4. Open the `.env` file in this folder and replace:
   - `SMTP_USER` with the Brevo SMTP login
   - `SMTP_PASSWORD` with the Brevo SMTP key
   - `SMTP_EMAIL` with your verified sender email address
5. Save `.env`, install requirements (`pip install -r requirements.txt`) and run `python app.py`.

EcoGrid automatically loads `.env` at startup. The SMTP host is `smtp-relay.brevo.com`, port `587`, with STARTTLS. Keep `.env` private and never upload it to GitHub or share its contents. Every new user receives their own OTP at the email address entered in the EcoGrid registration form. The account is created only after the OTP is verified.

Official Brevo instructions:
- SMTP relay: https://help.brevo.com/hc/en-us/articles/7924908994450-Send-transactional-emails-using-Brevo-SMTP
- SMTP keys: https://help.brevo.com/hc/en-us/articles/7959631848850-Create-and-manage-your-SMTP-keys
