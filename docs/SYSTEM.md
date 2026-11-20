# Smart Sensor Monitoring Hildesheim: the final predictive maintenance system

This document covers components 1 to 10 of the final system. It explains, for each component, the method and why it was chosen, the mathematics, the results actually obtained, how the sensor families differ, how useful it is, its limits and how it connects to the rest. The data foundation and Module 1 have their own documents: `README.md` and `docs/MODULE1.md`.

**Evaluation protocol for all components.** Folds run forward in time, month by month. Model choice uses the validation months, August and September 2026. The held-out test months are October 2026 to 18 November 2026, and they were touched only once. Every feature at an hour end is computed from information available at that moment, and tests check this by truncating the data. Confidence intervals come from a block bootstrap over devices. Where a component was changed *after* its test or replay output had been inspected, the change is listed under "Post-inspection revisions" at the end. The figures affected by those changes are labelled in-sample for that decision.

## The ten questions and where they are answered

| question | component | output (dashboard / e-mail / table) |
|---|---|---|
| 1. Is the sensor OK? | Module 1 state + overall light (9) | `state`, `traffic_light`, `overall_light` (worst of communication, battery, anomaly, risk) |
| 2. When might it go down? | 1 interruption risk | `risk_long` (24 h; 72 h for slow families), `risk_prolonged_7d`, `p_silence_becomes_outage`, `p_still_down_24h/7d` |
| 3. Is the gateway OK? | 2 gateway | `condition` per site and hour |
| 4. When might the gateway go down? | 2 gateway | empirical interruption rate with 90 % CI; **no predictive model** (warning signals tested and not predictive) |
| 5. Battery condition now? | 3 battery | `battery_condition` per chemistry; `not_measurable` where the family has no telemetry |
| 6. When does the battery become critical? | 3 battery | probability of crossing the threshold within 30 d and crossing-date quantiles, **only for lead-acid voltage series** |
| 7. Are the measurements reliable? | 4 anomalies | rule layer + de-biased neighbour residual CUSUM + static offsets + frozen runs |
| 8. Probable cause? | 5 diagnosis | `probable_cause`, `cause_confidence`, `cause_evidence`, `alternative_causes` |
| 9. Who to inspect first? | 6 prioritisation | "Diese Sensoren jetzt prüfen" list, `priority_score`, `why`, `recommended_action` |
| 10. Automatic e-mail | 7 notifications | incident engine: open, reminders, escalation and recovery, grouped by common cause |

---

## 1. Sensor interruption risk (`hsm/risk.py`, `run_risk.py`)

**Tasks.**
- **A:** onset of an interruption within H hours, from a working state. H is 6 h and 24 h; for Agvolution and Bigbelly it is 24 h and 72 h. A also covers a *prolonged* interruption of 24 h or more within 7 or 14 days.
- **B:** whether a developing silence becomes a declared outage.
- **C:** detection delay.
- **D:** while an outage lasts, whether the sensor is still unavailable 24 h or 168 h later.
- **E:** chronic outage rates.

**Candidates** (per family and task; the choice is made on validation):
- `family_rate`: Poisson rate of the family, P = 1 − e^{−λH}.
- `device_hazard`: gamma-Poisson shrinkage, λ̂_i = (k_i + κ λ_f) / (e_i + κ) with prior exposure κ = 168 h.
- `logit_shared`.
- `gbm_shared`: histogram gradient boosting with family identity as a categorical feature. Device weights are 1/√n_i, so dense families do not dominate.
- `gbm_family`: only where a family has at least 8 devices and 100 episodes.
- For B: silence-length rate and HMM surprise with isotonic calibration.
- For D: Kaplan–Meier (pooled / family / family × scope).

**Targets and censoring.** A label exists only if its outcome window closes before the export end of the family's table. Windows that overlap a fold boundary are purged.

**Results** (test months; AUC / PR-AUC / base rate; event recall at the threshold chosen on validation):

| family | A 6 h | A 24 h | A prolonged 7 d | B developing | D 24 h / 168 h |
|---|---|---|---|---|---|
| MeteoHelix | 0.80 / 0.05 / 0.009, recall 76 % | 0.80 / 0.21 / 0.030 | 0.92 (2 events, not usable) | 0.94 / 0.47 | 0.98 / 0.98 |
| MeteoWind | 0.80 / 0.06 / 0.011 | 0.81 / 0.30 / 0.039 | no test events | 0.92 / 0.34 | too few |
| Sensoterra | 0.59 / 0.06 / 0.013 | 0.59 / 0.13 / 0.048 | **0.87 / 0.38 / 0.049**, event recall 53 %, median lead 139 h, 1.17 false alert episodes per sensor-week | 0.92 / 0.61 | 0.79 / 0.90 |
| Ott, Bigbelly, Dragino, Adeunis, Clover, Agvolution | no or too few test events: no accuracy is claimed | | | | KM only |

- **Shared versus family models.** Shared models won wherever both were possible; `gbm_family` was never selected.
- **Sensoterra, short horizon.** The 6 h and 24 h onset of a Sensoterra outage is barely predictable (AUC about 0.59). Its prolonged outages are predictable a week ahead, because they follow recognisable precursors: repeated short outages and a falling onset-rate ratio.
- **Task B** separates real outages from bursts well: 148 of 156 Sensoterra declared outages had an alarm, a median 3.5 h before declaration. This early warning is **off by default**, because it costs about 1.4 false alerts per sensor-week.
- **Task C,** median delay from onset to declaration: Sensoterra 13.5 h, MeteoHelix and MeteoWind about 1.1 h, Clover 0.6 h, Agvolution 42 h (4 h reporting interval).
- **Onset hour pattern:** MeteoHelix at 01 and 07 UTC, MeteoWind at 05–07 UTC (χ² p < 1e‑17). This is a site or schedule effect worth checking.

**Usefulness.**
- Weekly planning, through the Sensoterra 7 d risk.
- Triage of developing silences (task B).
- An estimate of outage duration (task D) inside the down-sensor e-mail.

**Limitations.**
- No physical failure labels exist, so the target is *communication* interruption.
- Several families have no test events.
- Probabilities are calibrated (ECE 0.006–0.028) but low in absolute terms.

**Deployment.** `hsm/production.py` refits the selected model types on all labelled data. The models are shared GBMs for A 24 h, A prolonged 7 d and B, plus Kaplan–Meier per family for D. Scoring happens at each monitoring cycle.

## 2. Gateway monitoring (`hsm/gateway.py`)

The exports contain no gateway health telemetry, so a gateway is observed *indirectly* through the receptions it forwards.

- **Expected activity.** For each site s and hour h, the expected record count is μ_{s,h}, the trailing 168 h mean, **frozen** during inactive runs so that the baseline does not decay. P(0 records | μ) = e^{−μ}.
- **Conditions:**
  - `inactive_gateway`: zero records with P0 < 0.01 while its dependent sensors are heard elsewhere (re-routing, cumulative maximum within the run).
  - `inactive_unexplained`: zero records and the dependants silent too.
  - `network_or_platform`: 70 % or more of active sites silent.
  - `reduced_activity`.
  - `low_traffic`.
  - `insufficient_history`.
- **Events:** a run of 2 hours or more.

**Results.**
- 28 events: 10 network-level (including 29 Sep 08–10 UTC), 17 gateway-specific, 1 unexplained.
- Site `unlisted_52.1528_9.9570` has been silent since 5 Nov, for 281 h at the export end, and its sensors were re-routed.
- 30 of 114 sensors depend on a single gateway: if it fails, they have no alternative path.
- **Predictability.** A permutation test of four warning signals in the 24 h before events found nothing usable: activity ratio, device ratio, SNR shift and reduced activity all gave AUC 0.44–0.50, p ≥ 0.06. **Gateway outages are therefore not predicted.** The system reports an empirical gamma-Poisson interruption rate per site with a 90 % interval, plus the dependency count. A predictive accuracy for gateways would have been fabricated.

## 3. Battery health (`hsm/battery.py`)

**Signal per chemistry** (`config.BATTERY_OBSERVABILITY`). Voltage is **never converted to a percentage**.

- **Lead-acid** (Bigbelly 12 V SLA; Ott 12 V solar):
  - The daily minimum serves as the resting-voltage proxy.
  - Conditions use generic lead-acid resting references: good at 12.3 V or more, fair 12.0–12.3 V, low 11.7–12.0 V, critical below 11.7 V. Sources: OutBack, Samlex, Dynamis. These are generic for the chemistry and not manufacturer data for these exact devices, as `config` documents.
  - Three forecast models are back-tested on the October–November test period: persistence, Theil–Sen, and a local-linear-trend state-space model. The model is chosen per family on validation.
  - P(crossing) = max_k Φ((v* − μ_k)/σ_k) for k ≤ 30 d. Crossing dates are the first k at which this reaches 10 / 50 / 90 %.
- **Lithium** (MeteoHelix, MeteoWind, Agvolution, Dragino): the voltage plateau is flat, so only step changes and error codes are reported. **No crossing date is estimated.**
- **Clover:** the manufacturer's level estimate, with its trend; no depletion date.
- **No telemetry:** `not_measurable`, with no estimate.

**Results.**
- Bigbelly declines 19–27 mV per week. Device 500300550 is at 11.84 V, condition low, with P(below 11.7 V within 30 d) ≈ 0.10.
- Ott is stable.
- Test MAE:
  - Bigbelly: LLT 16 mV at 7 d and 23 mV at 14 d, with 80 % interval coverage of 0.88 and 0.79. Persistence: 38 and 62 mV.
  - Ott: persistence is best (21 and 31 mV), and the trend models add nothing.
- Clover reads 94–99 % and falls about 0.5 % per week. No depletion date is given, because a linear extrapolation of a manufacturer estimate is not a battery model.

## 4. Measurement anomalies (`hsm/anomaly.py`)

- **Rule layer:** manufacturer ranges, censored values, frozen runs (calm wind excluded), channels never valid, error codes. Each rule carries a severity.
- **Statistical layer for weather stations:**
  - The residual against leave-one-out neighbours, r_t = x_t − median_{j≠i} x_{j,t}.
  - It is de-biased by the station's own median over the trailing 28 days (causal).
  - **CUSUM** S⁺_t = max(0, S⁺_{t−1} + r_t − k), with k = sensor accuracy and threshold h = 24 × accuracy.
  - EWMA (λ = 0.02) and a MAD spike detector.
  - Persistent offsets are reported separately as *static offsets*. A pressure offset is interpreted as altitude, not as a fault.
- **Soil:** robust jump detection.

**Evaluation by synthetic fault injection,** because no labelled faults exist. Faults are injected into clean test periods, scaled to the sensor accuracy.
- CUSUM detects 100 % of biases of 2.5× accuracy or more, and of drifts of 5× accuracy per week.
  - Pressure bias: median delay 16 h at 2.5×, 6 h at 5×. RH bias: 13 h and 5.5 h.
  - False alarms on clean data: 0 per station-week for pressure, 0.54 for RH.
- Isolation Forest was trained before the test start. It works for pressure (95–100 %, 0.24 false alarms per week), but it **fails for RH** (0–7 %) and for temperature, so it was not adopted.

**Real findings.**
- Station `ac1f09fffe0e23a2`: −0.75 °C, +3.8 % RH and −8.5 hPa against its neighbours. This looks like siting.
- Four pressure offsets are consistent with altitude.
- A frozen soil temperature on Sensoterra probe …2961.

## 5. Probable cause (`system.diagnose`)

The diagnosis is a transparent evidence model. Rules are applied in order of evidence strength: the first rule that matches gives the cause, and later matches are kept as alternatives. Confidence is the strength of the evidence, not a calibrated probability.

| evidence | cause | confidence |
|---|---|---|
| uplinks heard after the last stored measurement | backend storage failure | high |
| most gateway sites silent in the same hour | network / platform | high |
| primary gateway inactive | gateway unavailability | medium |
| improbably many devices of one family stopped together | fleet-wide stop | medium |
| Sensoterra daily mode or majority of readings = 0 | soil contact / installation | medium |
| missing, invalid or frozen channel | measurement-channel failure | medium |
| resting voltage in the low or critical band | low battery charge | medium |
| receptions at or beyond the SF12 reference limits | weak radio | medium |
| only this device silent | individual failure (undetermined) | low |

Every e-mail and dashboard view shows the evidence and the alternatives. No probability is attached to a cause, because there are no ground-truth labels to calibrate one against.

## 6. Maintenance prioritisation (`system.priority`)

score = family importance × (severity of the overall light + 12·log(1 + hours unavailable) + 4·log(1 + expected missing reports) + 20·risk + battery and anomaly terms).

- All weights are in `config.PRIORITY_WEIGHTS`, an *operational* setting meant to be tuned by the operator.
- Every row states `why` and `recommended_action`.
- Gateways are ranked by down-time and number of dependants.

## 7. Automated e-mail notification (`hsm/notify/`)

**Incident life cycle** (SQLite store, file name from `notifications.store`):
1. **OPEN** when the condition becomes true.
2. **REMINDER** after 24 h for critical incidents and 72 h for warnings. The interval doubles after every notification, up to a cap of 168 h.
3. **ESCALATION** of a critical incident that is still unacknowledged after 48 h, sent to the escalation recipients.
4. **RECOVERY** after the condition has been clear for a minimum time that depends on the type: SENSOR_DOWN 2 h and at least 6 report intervals, backend failure 12 h, and 24 h for a flapping sensor (3 or more onsets in 7 days).

Further rules:
- If a condition returns within the 6 h cooldown, the incident is reopened silently, with no duplicate.
- A risk warning clears only once the risk is below 0.7 × its threshold (hysteresis).
- Sensors without information are frozen rather than recovered.
- **Grouping:** OPEN and RECOVERY events that share a cause (network, gateway site, family backend, fleet, data feed) go out as one e-mail listing every affected device.
- **Data feed:** an export stream that stops being refreshed opens a single `FEED_STALE` incident. Its sensors become "insufficient information" instead of 150 false outage alarms. Whether a stream is stale or the silence is real is decided from the file's modification time.

**Outage incidents do not depend on predicted risk.** SENSOR_DOWN, BACKEND_FAILURE, GATEWAY_DOWN and NETWORK_OUTAGE come only from the observed state. Tests check both directions: an outage with zero risk is notified, and a high risk never opens an outage incident.

**Security.**
- SMTP credentials are read only from environment variables (`HSM_SMTP_HOST/PORT/USER/PASSWORD/FROM`) and are never stored in the config or the code. A test scans the source for hard-coded credentials.
- Connections use STARTTLS, with up to 4 attempts and exponential back-off.
- By default `enabled: false` and `dry_run: true`, so e-mails go to the outbox as `.eml` files. Real delivery needs both `enabled: true` **and** `dry_run: false`.
- Replay, evaluation and tests force dry-run in code.

**Templates** (`templates/email.{txt,html}.j2`). Each e-mail contains:
- device, family, manufacturer and model;
- condition and severity;
- last reception and last stored data;
- reporting interval and current mode;
- time unavailable and missing reports;
- risks, battery and gateway;
- cause, confidence and evidence;
- recommended action and a dashboard link.

## 8. Historical replay (`hsm/replay.py`)

The full system runs hour by hour from 1 Aug to 18 Nov 2026, using only causal inputs, with dry-run delivery.

**Results.** These figures are **in-sample for the notification policy**, which was tuned on this replay; see the revisions at the end.
- 3,244 e-mails: 927 open, 1,164 reminders, 177 escalations, 976 recoveries. That is 1.33 per device-week.
- 18 e-mails were grouped. There were no duplicates, and the largest number of e-mails for one incident was 19.
- **Outages of 6 h or more:** 100 % of those still ongoing at an hourly check were notified (389 of 389). Overall 96.3 % were notified; the rest recovered within their declaration hour. The median notification delay after onset is 9.6 h, which is mostly the causal declaration delay of Sensoterra.
- **Premature recoveries** (the condition returned within 24 h): 10.8 % overall, 13.7 % for SENSOR_DOWN, 44 % for BACKEND_FAILURE (the backend failures were intermittent).
- **Risk warnings:** 32, of which 69 % were followed by a prolonged outage within 7 days. Only **14 % of prolonged outages** were warned of in advance. The warning threshold met the false-alert budget only for Sensoterra.

**Demonstrations** (`demo_emails/*.eml`, `demonstrations.parquet`), with what the system knew at the time and what happened afterwards:
- the Sensoterra backend failure of 13 Nov, as one grouped e-mail for 12 probes, cause "backend storage failure" with high confidence;
- the Dragino backend failure;
- the network outage of 29 Sep;
- a Clover fleet stop;
- gateway 52.1528 down;
- low battery on a Bigbelly;
- the frozen temperature;
- the station bias;
- a predictive warning 18 h before a 348 h outage;
- Sensoterra switching to its reduced daily mode.

## 9. Web app and dashboard (`hsm/webapp/`, `hsm/dashboard/`)

**Interactive web app** (Flask, Leaflet map, Chart.js; all libraries bundled). It is written for operators and repair staff, not for data scientists. Every text is a translation of a model output in `hsm/webapp/plain.py`; nothing new is inferred.

- **Map.** Sensors sit at their GPS positions from the data. Each has a colour for its overall condition, a symbol for what it measures (💧 soil moisture, 🌡️ weather or temperature, 🌬️ wind, 🌊 water station, 🗑️ waste bin) and an animated battery badge with the measured value. Markers are clustered: a cluster takes the colour of its worst member and shows how many sensors need action. Gateways appear as 📡 tiles. Street and satellite backgrounds are available.
- **Sensor panel**, opened by clicking a sensor:
  - status in one sentence;
  - "What is the problem?" (the cause and how certain it is);
  - "👉 what to do", and whether a site visit is needed;
  - battery: widget, trend and 30-day forecast;
  - last contact;
  - "Looking ahead": the AI forecasts as Low, Medium or High with a percentage and a note on how well that forecast was tested for this sensor type;
  - measurement checks;
  - 120-day charts: data received, measurements, battery, breakdown risk;
  - past breakdowns and the e-mails sent (full text viewable);
  - a collapsible technical section for engineers.
- **🧭 Directions** opens Google Maps in a new window with the route to the sensor. For the 25 sensors without a GPS position in the data, the route goes to the area of their main gateway, and the panel says so. **No position is ever invented**: these sensors are listed separately and are not placed on the map.
- **Battery badge.** 12 V batteries show the voltage and a band (good, fair, low or critical), not a percentage. CLOVER shows the level the sensor itself reports. Lithium batteries show "OK" or "Check", because a flat voltage carries no level. Sensors without battery telemetry show "?".
- **Other tabs:**
  - "Check these first" (Diese Sensoren jetzt prüfen): ranked cards with the problem, the action, a site-visit badge and directions;
  - alerts and e-mails, with each sent e-mail viewable;
  - gateways, with historical interruption frequency (gateway outages are not predicted);
  - "How good are the predictions?": the test results in plain words, with real examples from the history;
  - help.
- **Time slider and ▶ playback.** These show the network exactly as the system saw it at any hour since April 2026, using causal snapshots.
- **Live mode.** A background thread runs the monitoring cycle; the map refreshes itself, and "Check now" starts a cycle immediately.

The static single-file dashboard (`python -m hsm.cli dashboard`) remains available for offline sharing.

## 10. Deployment

```
pip install -r requirements.txt
python -m hsm.cli build-foundation --raw <exports> --out foundation
python -m hsm.cli analyse  --foundation foundation          # components 1-4 + walk-forward evaluation
python -m hsm.cli train    --foundation foundation --models models
cp replay/warning_thresholds.json models/                    # after the replay (thresholds chosen on validation)
python -m hsm.cli replay   --foundation foundation --out replay
python -m hsm.cli prepare  --foundation foundation --replay replay --out appdata
python -m hsm.cli app      --data appdata [--live-config config/hsm.yaml --models models]   # http://localhost:8050
python -m hsm.cli dashboard --foundation foundation --replay replay --out dashboard.html      # static alternative
python -m hsm.cli notify-test --config config/hsm.yaml       # dry-run unless enabled
python -m hsm.cli monitor --config config/hsm.yaml --models models [--once]
python -m hsm.cli ack --config config/hsm.yaml --incident 42
pytest -q tests
```

**A live cycle** (`monitor`) runs these steps:
1. Checks each export stream for staleness.
2. Rebuilds the foundation with observation end = *now* for current streams. A sensor that has gone silent therefore keeps accumulating missed reports even when no new data arrives.
3. Recomputes the gateway, battery and anomaly components.
4. Scores with the production models.
5. Runs the incident engine for the status hour.
6. Writes the live state and rebuilds the dashboard.

On 2 vCPU a cycle takes about 12 minutes, because the foundation is rebuilt from the full history. Polling therefore defaults to every 30 minutes, and the worst-case notification latency is the poll interval plus the cycle time plus the causal declaration delay. Incremental ingestion is the obvious next step.

The system needs no live sensor API. Real e-mail needs credentials in the environment. Neither was available here, so every e-mail in this delivery was a dry run.

---

## Post-inspection revisions (disclosed)

These changes were made **after** looking at validation, test or replay outputs. The Module 1 held-out results were not changed; they are in `docs/m1_evaluation.json`.

1. **Notification policy:**
   - reminder back-off;
   - recovery clearing that depends on the type and on flapping;
   - risk hysteresis;
   - the grouped subject line.

   These were changed after the first replay produced too many reminders and premature recoveries. The replay figures are therefore in-sample for these settings. The outage-coverage figures do not depend on them, because outage incidents open from the observed state.
2. **Warning thresholds:** the risk-warning threshold uses a false-alert budget of 0.25 episodes per sensor-week, chosen on the validation months. Families that cannot meet the budget get no warnings.
3. **Gateway:** μ is frozen during inactive runs, and re-routing is the cumulative maximum within a run. Both were fixed after a long outage showed a decaying baseline and dependants being reassigned.
4. **Anomalies:**
   - de-biasing by the station's own trailing median, after static offsets swamped CUSUM;
   - calm wind excluded from the frozen-value rule;
   - injection magnitudes expressed in multiples of the accuracy.
5. **Module 1 state:** short history is treated as insufficient information *after* the backend rule, because a Dragino backend failure had been hidden.
6. **Dragino** uses the hierarchical default, because it has too few devices for any supervised model.
7. **Live mode:** stream-level feed staleness and timeline-based degraded duration were added after the first simulated live cycle had flagged every sensor down on a stale export.
8. **E-mail policy (20 Nov 2026, at the user's request):**
   - E-mails go out only for "sensor stopped working" and "may stop within 24 hours". See `docs/EMAIL_ALERTS.md`.
   - The 24-hour alert level uses `EMAIL_BUDGET`: at most 0.10 false alerts per sensor-week and at least 10 % recall, both on the validation months.
   - Test-month metrics were visible while this budget was chosen, so the 24-hour e-mail figures are not a clean held-out result.
   - Replay with this policy: 384 e-mails; 361 of 361 outages where the sensor itself stopped were e-mailed; 46 % of 24-hour warnings were followed by a breakdown within 24 h.
9. **Backend-state latch:**
   - A device heard once after its last stored record, then silent for months, stayed "data not saved". It is now "not sending" unless it was heard within max(24 h, 6 report intervals) (`system.correct_backend_latch`).
   - The correction is applied downstream of Module 1; the Module 1 timeline and evaluation are unchanged.

## Main limitations
- There are no physical failure labels and no gateway telemetry. All predictions concern communication interruptions observed in the exports.
- Several families have no test events, and no accuracy is claimed for them.
- Gateway outages are not predictable from these data.
- Battery thresholds are generic for the chemistry, and lithium devices get no crossing date.
- The diagnosis rests on rules. Its confidence is evidence strength, not a probability.
- The notification policy is tuned in-sample.
- The live cycle rebuilds everything, so it takes minutes, not seconds.
