# Hildesheim Smart Sensor Monitoring

Predictive maintenance for the Hildesheim LoRaWAN sensor fleet: communication state, interruption risk, gateway monitoring,
battery health, measurement anomalies, probable cause, maintenance priority, automatic e-mail notification, historical replay
and a dashboard. Method, mathematics, results and limitations of every component: [`docs/SYSTEM.md`](docs/SYSTEM.md).

## The web app (start here)
An interactive map of the whole network for operators and repair staff:
- Sensors are shown at their real GPS positions, coloured red, yellow or green, with a symbol for what each one measures and an animated battery badge.
- Clicking a sensor opens a plain-language panel: what is wrong, what to do, whether a site visit is needed, the battery, the AI forecasts, the measurement checks, history charts and the e-mails sent. A 🧭 button opens the route in Google Maps in a new window.

```
pip install -r requirements.txt
python -m hsm.cli app --data appdata          # then open http://localhost:8050
```
Or double-click `start_app.bat` (Windows) or `start_app.sh` (macOS / Linux). To host it on a server: `docker build -t hsm . && docker run -p 8050:8050 hsm`.

The `appdata/` folder is produced by `python -m hsm.cli prepare --foundation foundation --replay replay --out appdata`. It is already included in the delivery, so the app runs straight away. The map background needs an internet connection (OpenStreetMap / Esri tiles); everything else runs offline.

**Live operation.** Run `python -m hsm.cli app --data appdata --live-config config/hsm.yaml --models models --host 0.0.0.0`. Every poll interval the app re-reads the exports, runs all models, sends the e-mails (dry-run unless enabled) and refreshes the map. A "Check now" button starts a cycle at once.

## Quickstart (full pipeline)
```
pip install -r requirements.txt
cp config/hsm.example.yaml config/hsm.yaml                    # edit paths and recipients; no secrets in this file
python -m hsm.cli build-foundation --raw <exports> --out foundation
python -m hsm.cli analyse   --foundation foundation            # Module 1, risk, gateway, battery, anomaly + evaluation
python -m hsm.cli replay    --foundation foundation --out replay        # causal replay, dry-run e-mails, demonstrations
python -m hsm.cli train     --foundation foundation --models models && cp replay/warning_thresholds.json models/
python -m hsm.cli dashboard --foundation foundation --replay replay --out dashboard.html --serve   # http://localhost:8050
python -m hsm.cli notify-test --config config/hsm.yaml        # test e-mail (dry-run unless enabled)
python -m hsm.cli monitor   --config config/hsm.yaml --models models          # live loop (--once for one cycle)
python -m hsm.cli ack       --config config/hsm.yaml --incident <id>          # acknowledge (stops escalation)
pytest -q tests
```
**Real e-mail** is sent only when `notifications.enabled: true` **and** `dry_run: false`, with credentials in the environment:
`HSM_SMTP_HOST`, `HSM_SMTP_PORT`, `HSM_SMTP_USER`, `HSM_SMTP_PASSWORD`, `HSM_SMTP_FROM`. Replay, evaluation and tests always run dry.

# Data foundation

Reproduces every table from the four original files (`devices_raw.csv`, `bodenfeuchte_raw.csv`, `weather_raw.csv`, `*Gateway_Standorte.csv`).

```
python run_foundation.py --raw <dir with the raw files> --out <output dir>            # full build (~11 min)
python run_foundation.py --raw ... --out ... --cutoff 2026-08-25                       # state of knowledge at a moment
FOUNDATION=<output dir> RAW=<raw dir> pytest -q tests                                  # verification suite
```
`manifest.json` records input SHA-256, code SHA-256, package versions, row counts, observation ends and the provenance of every constant.

## Layers
| layer | tables | rule |
|---|---|---|
| bronze | `bronze_{devices,soil,weather,gateways}` | every raw row verbatim as text + `raw_row_id` (`source:line`) + parsed `t` (UTC) |
| silver | `silver_{devices,soil,weather,gateways_official}` | typed, units harmonised, **flags instead of deletions**; duplicates marked `dup_rank` (0 first, >0 exact copy, -1 conflicting copy); failed backend inserts parsed (`ingestion_failed`, `error_class`, `error_detail`, `failed_batch_id`, recovered values `rec_*`); **record-level** flags (`flag_*`, decide `data_valid`) vs **field-level** clean columns (`*_clean`, `ec_us_cm`; raw kept) |
| gold | `gold_uplinks`, `gold_uplink_records`, `gold_gateway_sites`, `gold_radio_records`, `gold_silences`, `gold_reception_gaps`, `gold_hmm_fits`, `gold_network_hours`, `gold_site_hour`, `gold_device_hour` | causal derivations; every uplink links to its raw records |

## Availability definitions (all bounded in [0, 1])
Expected reports are **slots**; an arrival fills at most one slot; arrivals within half an interval of the previous filled slot are **excess reports** (counted separately).

| measure | arrivals counted | schedule | meaning |
|---|---|---|---|
| `service_availability` | uplinks whose data was **stored** in the family's measurement table | declared nominal | primary KPI: was the expected data delivered? |
| `valid_data_availability` | stored and not flagged corrupt / error code / outside the manufacturer range / stuck (record level; field-level problems such as censored GWC do not count) | declared nominal | usable data |
| `reception_availability` | any record (incl. radio metadata and failed inserts) | declared nominal | did the system hear the sensor? |
| `mode_availability` | stored data | reporting mode inferred causally (e.g. Sensoterra 24 h mode) | link reliability given what the sensor is trying to send |

Differences attribute causes: reception without storage = backend / data path; nominal vs mode = sensor in reduced reporting mode.

## Outage declaration (causal, an operational definition)
For each silence, `declared_at` is computed from information available when the silence began: the device's HMM2 fitted on the previous 28 days (weekly refit, excluding silences already declared), Beta estimate for reduced-mode slots, and the floor `max(24 h, 2 × mode interval)` (Agvolution 72 h, Bigbelly 48 h). `status`: declared / burst / undetermined. `scope_at_declaration` uses only declarations known at that moment; `scope_final` is the retrospective revision. `retro_annotation` is evaluation/descriptive only.

## Prediction eligibility (`gold_device_hour.eligible_for_prediction`)
Observable at the hour end: service started, inside the data table's observation window, at least 14 days of history, and not inside a **declared** outage. A silence that is developing but not yet declared stays eligible and is marked `developing_silence`. Whether a label can be scored (outcome observable before the export end) is an evaluation rule, applied in the evaluation module, never in eligibility.

## Constants
All constants live in `hsm/config.py` with provenance (`external`, `declared`, `physical`, `structural`, `operational`, `exploratory`, `retrospective`) and permitted use. Removed full-history constants: Sensoterra 24 h-mode loss 0.091 (now causal Beta estimate), family-median loss fallback (now Beta prior), gateway sites discovered from the full data (now causal site registry), HMM fitted on full history (now weekly causal fits).

## Manufacturer-informed rules (v2, 2026-11-20)
Every constant now carries `sources` (ids in `config.SOURCES`) and a `verification` level; external constants must have a source (enforced at import). Full rule table, effects and source URLs: [`docs/MANUFACTURER_SPECS.md`](docs/MANUFACTURER_SPECS.md). Changes against v1:

| change | columns |
|---|---|
| soil temperature validity uses the manufacturer range; > 40 C inside the range = exposure (valid) | `flag_soil_temp_out_of_spec` (replaces `flag_soil_temp_implausible`), `flag_soil_temp_exposure_suspected`, `soil_temp_range_source` |
| CLOVER GWC = 0 is censored (dry end of the conversion table) | `flag_gwc_censored_dry`, `gwc_clean` |
| CLOVER remaining days above the rated 11-year life are impossible (bound was 20000, exploratory) | `flag_remaining_days_implausible`, `battery_remaining_days_clean` |
| Dragino EC in uS/cm; unscaled whole-number glitches | `ec_us_cm`, `flag_ec_suspected_unscaled` |
| MeteoWind direction undefined at calm | `flag_wind_calm`, `wind_dir_clean` |
| Adeunis -99 temperature | `device_temp_c_clean` |
| LoRa link-margin and saturation flags; families without radio metrics | `gold_uplinks.flag_snr_below_sf12_reference_floor`, `flag_rssi_below_sf12_reference_sensitivity`, `flag_beyond_sf12_reference_limits`, `flag_rssi_near_saturation`, `radio_metrics_available` (same flags per record in `silver_devices`) |
| documented meaning of the Sensoterra 24 h mode | `gold_device_hour.svc_mode_meaning` |
| battery signal per family (no voltage-to-percent conversion) | `config.BATTERY_OBSERVABILITY` |

## Foundation v3 fixes (2026-11-20)
- **Span-anchored uplink merging:** no chained uplinks beyond 60 s.
- **Service ledger timing:** uses `t_data`, the time the first stored measurement record became available.
- **Stored data must carry a measurement** (`stored_without_measurement` flags metadata-only uplinks).
- **Strict channel validity:** a causal channel registry (Dragino ch1–4, Agvolution profile/cap_air) defines the expected channels; `data_valid` requires all of them, `data_valid_any` at least one.
- **Reporting mode from reception:** the service ledger's reporting mode is the mode inferred from reception (what the device sends), so storage failures cannot fake a mode.
- **Renamed flags:** `flag_beyond_sf12_reference_limits` (link margin cannot be established), `flag_soil_temp_exposure_suspected`, `flag_ec_suspected_unscaled`.

## Module 1
See [`docs/MODULE1.md`](docs/MODULE1.md).
