# Manufacturer-informed rules in the data foundation

Sources were accessed on 2026-11-20. Each rule names the constant in `hsm/config.py` that holds it, the source IDs (`config.SOURCES`, URL table at the end), and a **verification** level:

| level | meaning |
|---|---|
| `verified` | the document describes our device model and the property in question |
| `sibling_model` | the document describes a closely related model from the same manufacturer (e.g. Dragino LSE01 for SE0X-LB, the Sensoterra multi-depth sheet for single-depth probes) |
| `general_family` | the document describes the product line, but our exact model is not identified (OTT, Bigbelly, Adeunis), or it is a generic LoRa reference |
| `unverified` | rule derived from the data, no manufacturer statement found |

Rules whose value was **derived from this dataset** keep provenance `exploratory`, even when a manufacturer document supports how they are interpreted.

## Principles

1. **Raw values are never overwritten.** Every rule adds a flag or a `*_clean` column next to the raw column.
2. **Two levels of validity.**
   - *Record level* (`flag_*` → `gold_uplinks.data_valid` → `valid_data_availability`): the whole record is corrupt, outside the manufacturer range, or stuck.
   - *Field level* (`*_clean`): a single field is unusable (sentinel, censored, undefined, glitch). The record still counts as valid delivered data.
3. **Physical site limits beat manufacturer ranges only where the physics is certain.**
   - Air pressure at Hildesheim altitude and air temperature in Germany keep the stricter physical bounds (`PLAUSIBLE`).
   - For soil temperature the physics is not certain: an exposed probe reads its own case temperature. So the manufacturer range decides validity, and readings above 40 °C are flagged as exposure.
4. **No battery percentage or remaining life is derived from voltage.** No manufacturer discharge curves were found.

## Rules

| Constant | Rule | Sources | Verification | Effect in the foundation |
|---|---|---|---|---|
| `NOMINAL_MIN` | Default intervals: MeteoHelix 10 min, CLOVER 15 min, Dragino 20 min, Sensoterra 24 uploads/day | BARANI_HELIX_DECODER, TEKTELIC_CLOVER_QSG, DRAGINO_SE0X_MANUAL, SENSOTERRA_SD_TECHSHEET | verified | unchanged values; Ott, Agvolution, Bigbelly and Adeunis remain *declared from the data* |
| `ALT_MODES_MIN`, `MODE_MEANING` | Sensoterra sending once a day while in soil = poor pin contact or too-dry soil | SENSOTERRA_FAQ_DAILY, SENSOTERRA_SD_TECHSHEET | verified | `gold_device_hour.svc_mode_meaning = reduced_daily_mode_contact_or_dry_soil` while the causally inferred mode is 24 h. Recommendation: re-seat the probe. Not a failure label |
| `SPEC_SOIL_TEMP_C` | Soil temperature range: Sensoterra −20…60 °C (operating environment, multi-depth sheet), CLOVER −15…60 °C, Dragino −40…85 °C (LSE01) | SENSOTERRA_MD_DATASHEET, TEKTELIC_CLOVER_SPEC, DRAGINO_LSE01_LORA_ALLIANCE | sibling_model | `flag_soil_temp_out_of_spec` replaces `flag_soil_temp_implausible` in record validity. `flag_soil_temp_exposure_suspected` marks readings in range but above 40 °C (valid). `soil_temp_range_source` says which bound applied (Agvolution: physical −15…40 °C) |
| `PLAUSIBLE` | Rain = difference of consecutive counter values (a reset gives one huge value); direct sunlight → CLOVER reports case temperature | BARANI_HELIX_DECODER, TEKTELIC_UG | verified | Bounds unchanged except `soil_temp_c` → `soil_temp_physical_c` (exposure only). The exploratory `clover_remaining_days_max` was removed |
| `CLOVER_RATED_LIFE_DAYS` | CLOVER rated battery life 11 years at 15-min reporting | TEKTELIC_CLOVER_SPEC | verified | `flag_remaining_days_implausible` now fires above 4,017.75 days (was 20,000); `battery_remaining_days_clean` |
| `BATTERY_OBSERVABILITY` | Which battery signal each family has, and its chemistry where documented | SENSOTERRA_SD_SHOP, TEKTELIC_CLOVER_SPEC, TEKTELIC_CLOVER_QSG, BIGBELLY_SPEC_18AH, BIGBELLY_SPEC_28AH, OTT_HYDROSYSTEMS, OTT_ECOLOG1000, ADEUNIS_TEMP_V3_UG, TTN_ADEUNIS_TEMP, AGVOLUTION_CLIMAVI | general_family | Input for the battery module. Sensoterra, MeteoHelix and MeteoWind: *not measurable*. CLOVER: level only, remaining days excluded |
| `WIND_CALM_MS` | Cup calibration m/s = 0.6335·Hz + 0.3582, so 0 means no rotation; direction undefined | BARANI_WIND_COMPACT, BARANI_WIND_DATASHEET | verified | `flag_wind_calm`, `wind_dir_clean` (NaN when calm) |
| `SPEC_ACCURACY` | MeteoHelix ±0.2 °C / ±1.5 % RH / ±1.5 hPa; MeteoWind 1 % or ±0.2 m/s, 2°; Sensoterra and CLOVER soil temperature ±2 °C | BARANI_HELIX_SHOP, BARANI_WIND_DATASHEET, SENSOTERRA_MD_TECHSHEET, TEKTELIC_CLOVER_SPEC | verified | reference tolerance for the anomaly module (e.g. MeteoHelix station …23f1 at −0.76 °C vs neighbours) |
| `EFFECTIVE_RESOLUTION` | Data steps: Sensoterra temperature 1 °C (fixed decimal per probe), vwc 0.1 % (= spec), rain 0.2 mm, Dragino EC 0.0001 | SENSOTERRA_MD_TECHSHEET, BARANI_HELIX_DATASHEET, DRAGINO_LSE01_LORA_ALLIANCE | sibling_model (exploratory value) | anomaly and trend modules must not resolve finer than these steps |
| `DRAGINO_EC_STORAGE` | EC stored in S/m (step 0.0001 = 1 µS/cm spec resolution); whole-number EC ≥ 1 = unscaled-value glitch | DRAGINO_LSE01_LORA_ALLIANCE | sibling_model (exploratory) | `ec_us_cm` (NaN for glitches and implausible values), `flag_ec_suspected_unscaled` (8 unique records, 30 Sep 06:24–08:04, directly before that device's failed-insert period) |
| `RADIO_LIMITS` | SNR floor −20 dB at SF12 (−7.5 dB at SF7); SF12 sensitivity about −137 dBm at 125 kHz; receiver saturation about −40…−30 dBm | SEMTECH_LORA_FAQ, LORA_SNR_TABLE, LORA_SENS_TABLE, SX1276_DATASHEET | general_family | `gold_uplinks.flag_snr_below_sf12_reference_floor`, `flag_rssi_below_sf12_reference_sensitivity`, `flag_beyond_sf12_reference_limits` (renamed from no-link-margin: link margin cannot be established without SF and gateway noise figure), `flag_rssi_near_saturation` (best reception of the uplink), plus the same flags per record in `silver_devices`. SF is not in the exports, so these are reference-limit flags, not link-margin estimates |
| `RADIO_METRICS` | Bigbelly is cellular; OTT stations use cellular/satellite; MeteoWind exports carry no radio metadata | BIGBELLY_HC5, OTT_HYDROSYSTEMS | general_family | `gold_uplinks.radio_metrics_available = False`; RSSI/SNR modules skip these families |
| `HMM_FAMILIES` | Ott and Bigbelly are not LoRa → floor-only outage rule | BIGBELLY_HC5, OTT_HYDROSYSTEMS | general_family | note updated, value unchanged |
| `FIELD_NEVER_VALID` | Adeunis temperature only ever carried the −99 sentinel (the device reports product-error alerts) | TAGO_ADEUNIS_TEMP | unverified | descriptive; record handling is `device_temp_c_clean` (NaN for −99) |
| `CLOVER` GWC (in `silver.soil`) | GWC is a table conversion of a frequency; in the data GWC = 0 exactly when the frequency is beyond the dry end of the table | TEKTELIC_FAQ | verified (mechanism) / exploratory (floor) | `flag_gwc_censored_dry`, `gwc_clean` (NaN when censored or rawfreq fault). Record stays valid |
| `OFFICIAL_GATEWAYS_FILE_GLOB` | gateway list supplied with the challenge | CHALLENGE_GATEWAY_LIST | verified | unchanged |
| `LEAD_ACID_REST_V` (battery component, added with the final system) | generic 12 V lead-acid resting-voltage references: about 75 % state of charge at 12.3 V, 50 % at 12.0 V, 25 % at 11.7 V. Open-circuit voltage needs hours to days to stabilise, so the night-time minimum of a solar-charged system is only a proxy | OUTBACK_MX60_SOC, SAMLEX_STANDING_V, DYNAMIS_SLA_MANUAL | general_family (not Bigbelly or OTT specific) | condition bands good / fair / low / critical and threshold-crossing targets for Bigbelly and Ott; never converted to a percentage |
| `LEAD_ACID_SOC_TABLE` (web app display, 21 Nov 2026) | approximate state of charge of a 12 V lead-acid battery: 12.6 V 100 %, 12.4 V 75 %, 12.2 V 50 %, 12.0 V 25 %, 11.7 V 0 % (Battery Council International); published tables differ by about 0.1-0.2 V | BCI_OCV_SOC, OUTBACK_MX60_SOC, SAMLEX_STANDING_V | general_family | shown as "about x %" for Bigbelly and OTT |
| `LITHIUM_FULL_V` / `LITHIUM_EMPTY_V` (web app display, 21 Nov 2026, user request) | rough charge of 3.6 V lithium cells by simple linear comparison: 3.6 V = 100 %, 3.0 V = 0 % | none (operational simplification) | assumption | shown as "about x %" for Agvolution, Dragino, Adeunis; the flat discharge curve means it stays near 100 % for most of the life and then drops fast; sensors without telemetry get no percentage |

## Not verifiable with public documentation

- **Agvolution:** interval, battery chemistry, meaning of the error codes 5889 and 7200.
- **OTT:** exact model behind the `anl_4509xx` IDs. The 12 V system voltage rules out the 3.6 V ecoLog 1000.
- **Adeunis:** model version. The observed 3.78–3.87 V is above the 3.6 V nominal of the V3 cell.
- **CLOVER:** unit of the raw frequency. TEKTELIC says kHz, but its conversion table is published only as an image.
- **Sensoterra:** temperature range and sensing depth of the single-depth probe. The multi-depth sheet was used instead.
- **Spreading factor:** not in the exports, so per-uplink link margin cannot be computed.

## Source list

| ID | Document | URL |
|---|---|---|
| CHALLENGE_GATEWAY_LIST | Gateway list supplied with the challenge | file://d06ac58c-Gateway_Standorte.csv |
| SENSOTERRA_MD_TECHSHEET | Sensoterra Multi Depth Tech Sheet v2.1 | https://res.cloudinary.com/dctlrnwuz/image/upload/v1670848753/Sensoterra_MD_Tech_Sheet_v2.1_EN_yczu5f.pdf |
| SENSOTERRA_MD_DATASHEET | Sensoterra Multi Depth datasheet (iot-shop.de) | https://iot-shop.de/shop/sensoterra-multi-depth-lorawan-bodenfeuchtesensor-4829/document/5466 |
| SENSOTERRA_SD_TECHSHEET | Sensoterra Single Depth Gen2.5 Tech Sheet | https://cdn.shopify.com/s/files/1/0447/5633/6807/files/Sensor_-_Sensoterra_SD_Gen2.5_Tech_Sheet_EN.pdf |
| SENSOTERRA_FAQ_DAILY | Sensoterra support: hourly data FAQ | https://servicedesk-ci.freshdesk.com/en/support/solutions/articles/80001186574-why-am-i-not-getting-hourly-data-from-my-sensor- |
| SENSOTERRA_SD_SHOP | Sensoterra Single Depth product page | https://iot-shop.de/en/shop/sensoterra-single-depth-bodenfeuchtesensor-4828 |
| TEKTELIC_CLOVER_SPEC | TEKTELIC CLOVER spec sheet (Mar 2026) | https://tektelic.com/wp-content/uploads/TEKTELIC_CLOVER-Agriculture_Sensor_Spec-Sheet_Mar-2026.pdf |
| TEKTELIC_CLOVER_QSG | TEKTELIC CLOVER Quick Start Guide | https://knowledgehub.tektelic.com/hubfs/Anuradha%20Kala%20QSG%20JULY%204/Clover%20Quick%20Start%20Guide.pdf |
| TEKTELIC_FAQ | TEKTELIC CLOVER/KIWI FAQ | https://knowledgehub.tektelic.com/clover-kiwi-faq |
| TEKTELIC_UG | TEKTELIC KIWI/CLOVER User Guide | https://tektelic.com/wp-content/uploads/TEKTELIC_KIW_UG.pdf |
| DRAGINO_SE0X_MANUAL | Dragino SE0X-LB user manual | https://wiki.dragino.com/xwiki/bin/view/Main/User%20Manual%20for%20LoRaWAN%20End%20Nodes/SE0X-LB--LoRaWAN_Soil_Moisture%26EC_Sensor_User_Manual |
| DRAGINO_LSE01_LORA_ALLIANCE | Dragino LSE01 LoRa Alliance device questionnaire | https://lora-alliance.org/wp-content/uploads/2021/11/LSE01-LoRaWAN-EU-Supplementary-Device-Info-Questionnaire-V1_0.pdf |
| BARANI_HELIX_DATASHEET | MeteoHelix IoT Pro DataSheet (2024) | https://barani.squarespace.com/s/MeteoHelix-IoT-Pro-DataSheet.pdf |
| BARANI_HELIX_DECODER | MeteoHelix message format | https://barani.squarespace.com/meteohelix-message-decoder |
| BARANI_HELIX_SHOP | MeteoHelix IoT Pro product page (accuracy) | https://iot-shop.de/en/shop/barani-meteohelix-iot-pro-wetterstation-5010 |
| BARANI_WIND_DATASHEET | MeteoWind IoT Pro DataSheet (2024) | https://barani.squarespace.com/s/MeteoWind-IoT-Pro-DataSheet.pdf |
| BARANI_WIND_COMPACT | MeteoWind Compact DataSheet | https://barani.squarespace.com/s/MeteoWind-Compact-DataSheet.pdf |
| ADEUNIS_TEMP_V3_UG | Adeunis TEMP V3 user guide | https://www.manualslib.fr/manual/278915/Adeunis-Temp-V3-Lorawan-863.html?page=38 |
| TTN_ADEUNIS_TEMP | TTN device repository: Adeunis Temp | https://www.thethingsnetwork.org/device-repository/devices/adeunis/temp/ |
| TAGO_ADEUNIS_TEMP | TagoIO: Adeunis Temperature | https://tago.io/devices/sigfox/adeunis-temperature.md |
| BIGBELLY_HC5 | Bigbelly HC5 station data sheet | https://www.procurement.iastate.edu/sites/default/files/Documents/Bigbelly%20Station%20Data%20Sheets%201901%20-%20HC5%20and%20SC55%20(1).pdf |
| BIGBELLY_SPEC_18AH | Bigbelly Smart Compost spec sheet | https://bigbelly.com/wp-content/uploads/2025/11/SmartCompostProductSpecSheet2507_WEB.pdf |
| BIGBELLY_SPEC_28AH | Bigbelly Smart Max product sheet | https://bigbelly.com/wp-content/uploads/2023/06/Smart-Max-Product-Sheet-2301.pdf |
| OTT_HYDROSYSTEMS | OTT HydroSystems | https://ott.com/products/system-solutions-23/ott-hydrosystems-384 |
| OTT_ECOLOG1000 | OTT ecoLog 1000 technical data | https://www.ott.com/products/water-level-1/ott-ecolog-1000-water-level-logger-2455/productAction/outputAsPdf |
| AGVOLUTION_CLIMAVI | Agvolution climavi brochure | https://iot-shop.de/shop/agvolution-climavi-wetterstation-6801/document/4269 |
| SEMTECH_LORA_FAQ | Semtech LoRa FAQ | https://www.semtech.com/design-support/faq/faq-lora/P0 |
| LORA_SNR_TABLE | Mobilefish LoRa tutorial 16 (SNR limits) | https://www.mobilefish.com/download/lora/lora_part16.pdf |
| LORA_SENS_TABLE | Blaszczyszyn et al., arXiv:1811.01886 | https://arxiv.org/pdf/1811.01886 |
| SX1276_DATASHEET | Semtech SX1276 datasheet | https://cdn.sparkfun.com/assets/7/7/3/2/2/SX1276_Datasheet.pdf |
| BCI_OCV_SOC | Battery Council International OCV / state-of-charge table, as quoted by Practical Sailor | https://www.practical-sailor.com/boat-maintenance/solving-the-power-puzzle/ |
| OUTBACK_MX60_SOC | OutBack Power MX60 manual: 12 V lead-acid voltage vs state of charge | https://auto.manualsonline.com/manuals/mfg/outback_power/mx60_1.html?p=85 |
| SAMLEX_STANDING_V | Samlex PST-300-12 manual, Table 7.4: standing voltage vs state of charge | https://www.manualslib.fr/manual/409057/Samplexpower-Pst-300-12.html?page=20 |
| DYNAMIS_SLA_MANUAL | Dynamis sealed lead-acid technical manual | https://manualmachine.com/dynamis/leadline/21505470-technical-manual |
