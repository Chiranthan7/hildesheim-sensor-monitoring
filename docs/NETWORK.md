# Radio network: focus mode, "Improve network" page and visit route

All values come from the measured radio records (`gold_radio_records`: which gateway received which message, RSSI, SNR),
built by `hsm/network.py` into `appdata/network.json` (also run by `prepare`). Nothing is simulated or invented.

## Signal classes (best gateway of each sensor, median SNR)
Limits from the Semtech SX1276/77/78/79 datasheet (LoRa demodulator SNR limit per spreading factor, SF7 −7.5 dB … SF12 −20 dB):

| Class | Median SNR | Meaning |
|---|---|---|
| Strong | ≥ 0 dB | large margin |
| Good | −7.5 … 0 dB | received even at the fastest data rate (SF7) |
| Weak | −15 … −7.5 dB | needs slow data rates (SF8–SF10): more airtime and battery |
| Very weak | < −15 dB | only SF11/SF12 work, close to the −20 dB limit; messages get lost |

A gateway "regularly" receives a sensor when it got ≥ 20 of its messages and ≥ 5 % of them.
Sensors without radio records (MeteoWind, OTT, Bigbelly, Dragino; OTT and Bigbelly use the mobile network) are shown as "no signal data".

## Distance effect (used by the new-gateway estimate)
Estimated **within sensors** heard by several gateways (device fixed effects), so antenna, burial depth and sensor type cancel out:
SNR changes by **−3.8 dB per tenfold distance** (SE 0.71 dB; 223 links of 74 sensors; residual SD 2.0 dB; within-R² 0.16).
RSSI shows no usable distance effect in this data (gateway sensitivity differences and adaptive data rate).
Only received messages exist in the data, so links too weak to deliver anything are missing; the true effect is probably
stronger, i.e. the estimate is conservative. Estimates are limited to 3.7 km (90 % of the measured links are shorter).

For a candidate gateway at distance *d* from a sensor whose best gateway is at *d₀* with median SNR *S₀*:
expected SNR = *S₀* − 3.8 · log₁₀(*d* / *d₀*).
* **Better signal**: weak or very weak today, expected gain ≥ 3 dB and a better class.
* **Backup gateway**: depends on one gateway today and the new one is expected to receive it at ≥ −7.5 dB.
* **Suggest best spot**: grid search (~75 m) maximising 2 × better + backup.

This is an empirical estimate, not a radio-coverage simulation. Buildings, vegetation and antenna height matter;
test a spot with a temporary gateway before installing.

## Buildings
2D building footprints from OpenStreetMap (Overpass API), loaded in the browser within 800 m of the new gateway.
For each line from the new gateway to a benefiting sensor the page counts the footprints it crosses (the buildings the gateway
and the sensor stand on are excluded). If the service is not reachable, nothing is counted and the page says so.

## Sensor focus mode
Selecting a sensor fades the other markers and draws a line to every gateway that regularly received it:
colour = signal class, thickness = share of its messages, dashed = less than 20 %.

## Visit route
Sensors that need a site visit (status "site visit needed", with a GPS position) are ordered by nearest neighbour + 2-opt
on straight-line distances, starting at the most urgent sensor or the user's location. Google Maps links are split into
parts of at most 10 points (Google Maps URL limit).
