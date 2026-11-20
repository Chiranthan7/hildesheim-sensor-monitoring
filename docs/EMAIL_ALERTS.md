# E-mail alerts

## When an e-mail is sent
Only in two situations:

| Alert | When | Based on |
|---|---|---|
| **Sensor stopped working** | the outage detector declares that a sensor has stopped sending (the silence has become statistically improbable for that sensor) | observed data only; never depends on a forecast |
| **Sensor may stop working within 24 hours** | the AI 24-hour forecast is above the alert level validated for that sensor type, **or** a 12 V battery is in the critical band (< 11.7 V resting voltage) | AI forecast (wind sensors, CLOVER) and measured battery voltage (OTT, Bigbelly) |

One e-mail per problem: no reminders, no escalation, no "fixed" e-mails. Sensors that stop together for the same reason (network outage, gateway off, same sensor type at the same time) are sent as **one** e-mail. When monitoring starts, problems that already exist are listed in **one** summary e-mail.
Everything else (server-side storage failures, low battery, unusual measurements, gateways) is shown in the app but not e-mailed.

Each e-mail contains:
- the sensor: what it measures, manufacturer and model, ID;
- its location: coordinates, the nearest gateway and a **Google Maps directions** button (gateway location, clearly marked, when the data has no sensor position);
- what was detected: since when no data has arrived, the last message, how the problem was detected, the AI forecast with how reliable it is for that sensor type, the probable cause and what to do.

Examples: `replay_results/example_alert_emails/` (open them with Mail).

## How the 24-hour alert level was chosen
For each sensor type, the alert level is the lowest level that, on the **validation months** (Aug–Sep), produced ≤ 0.10 false alerts per sensor per week **and** caught ≥ 10 % of breakdowns (`system.EMAIL_BUDGET`). Only wind sensors (MeteoWind) and CLOVER meet both criteria. For soil probes (Sensoterra) and weather stations (MeteoHelix), the 24-hour forecast was not reliable enough, so they get no 24-hour e-mails. They still get "stopped working" e-mails.

**Replay (Aug–Nov 2026, test mode, no real e-mails):**
- 384 e-mails in total, about 0.16 per sensor per week.
- Every outage where the sensor itself stopped was e-mailed: 361 of 361.
- 25 server-side failures (sensor heard but data not saved) were shown in the app only.
- 26 e-mails warned that a sensor might stop within 24 hours; 46 % were followed by a breakdown within 24 hours.

## Turning on real e-mails

Real sending needs `HSM_EMAIL_ENABLED=1` plus one of the two ways below. Recipients: `HSM_ALERT_TO=you@gmail.com` (several separated by commas). Credentials go only into environment variables, never into files in the repository.

### A. From your computer with Gmail (free)
1. Turn on **2-Step Verification** for the Google account, then create an **App Password** at https://myaccount.google.com/apppasswords (name it "Sensor monitoring"). Google shows a 16-character password; this is not your normal password.
2. In Terminal, inside the project folder (with `conda activate hsm`):
   ```
   export HSM_SMTP_HOST=smtp.gmail.com
   export HSM_SMTP_PORT=587
   export HSM_SMTP_USER=yourname@gmail.com
   export HSM_SMTP_PASSWORD=abcdefghijklmnop        # the 16-character app password, without spaces
   export HSM_SMTP_FROM=yourname@gmail.com
   export HSM_ALERT_TO=yourname@gmail.com
   export HSM_EMAIL_ENABLED=1
   "$CONDA_PREFIX/bin/python" -m hsm.cli notify-sample --config config/alerts.yaml
   ```
   Two e-mails marked [TEST] arrive within a minute: one "stopped working" and one "may stop within 24 hours", built from real cases in the data.

### B. From the online app on Render (free plan)
Render's free plan blocks e-mail (SMTP) ports, so Gmail's SMTP does not work there. Use **Resend**, which sends over normal web traffic:
1. Create a free account at https://resend.com **with the e-mail address that should receive the alerts**, and create an API key.
2. In Render, open the service, then **Environment**, and add:
   `HSM_RESEND_API_KEY` = the key, `HSM_ALERT_TO` = the same e-mail address, `HSM_EMAIL_ENABLED` = `1`, `HSM_ADMIN_TOKEN` = a code you choose.
3. In the app, open the **Alerts & e-mails** tab, enter the admin code and click **Send test e-mails**.

Without your own verified domain, Resend only delivers to the address of the Resend account itself. That is fine for alerts to yourself. (A paid Render instance, from 7 USD per month, allows Gmail SMTP as in A.)

## Automatic alerts (live monitoring)
Automatic e-mails need **new sensor data** to arrive. The data provided for the challenge is a fixed export that ends on 17 Nov 2026, so no new problems can occur in it.
- Once a live export or API exists, run the monitoring loop on any always-on computer: `python -m hsm.cli monitor --config config/alerts.yaml --models models`, with the export files in `./exports`. It checks every 30 minutes and sends the e-mails.
- Running it on Render needs a paid instance: the monitoring cycle needs more memory than the free plan's 512 MB.
