# Module 1: communication state and packet loss

`python run_m1.py --foundation <dir>` (about 6 min on the full data). Outputs are `gold_m1_*` tables plus `m1_evaluation.json`. Tests are in `tests/test_m1.py`: mathematics, purging, choice causality, state rules, reconciliation, truncation invariance and reproducibility.

## 1. Question and quantities
For every sensor and every hour end *h*, using only information resolved before *h*, Module 1 answers:

- **Current condition:** which communication state the sensor is in, with supporting evidence.
- **Expected future availability:** y_H = expected-report delivery over (h, h+H], and the expected number of missing reports.
- **Statistics:** per sensor and period, four availabilities kept separate:
  - **service:** stored measurement data, on the declared schedule;
  - **reception:** heard by the system;
  - **valid:** all expected channels stored and valid;
  - **mode:** delivered according to the reporting mode the device is in.
- **Packet loss** is defined for each of the four (= 1 − availability).
- Frame counters are not in the exports, so loss is measured against the expected report slots.

## 2. Mathematical formulation
**Slot process.** X_k ∈ {0,1} marks whether expected report *k* of the device's reporting mode was delivered.
- A missed slot is resolved at its deadline (due time + 0.5 interval).
- A filled slot is resolved when its data becomes available (`t_data`).

**Individual statistical models** (`hsm/m1/analytic.py`), trailing 28 days:
- Slots of a silence leave the *training* counts from the moment that silence is declared an outage. They stay in the *filter*.
- **Bernoulli:** p = (d+1)/(s+2).
- **Markov-1:** p_DD = P(X=1 | previous 1), p_MD = P(X=1 | previous 0), each with a Beta(1,1) prior.
  E[A_n] = (1/n) e_last · Σ_{h=1..n} P^h · [1,0]ᵀ.
- **EB-Markov (hierarchical):**
  - p = (n + κ·m)/(N + κ).
  - m is the mean of the family's *other* devices at the same hour.
  - κ comes from the method of moments: κ = m(1−m)/(Var_between − E[binomial variance]) − 1, bounded to [2, 5000].
- **HMM2:**
  - Hidden good/bad link state, refitted weekly on the trailing 28 days (foundation fits).
  - Forward filter: α_k ∝ (α_{k−1} A) ⊙ e(x_k).
  - Next-report probability: α A d. Horizon forecast: E[A_n] = (1/n) α Σ_h A^h d.
  - Silence surprise: S(k) = P(a silence lasts ≥ k reports | normal behaviour).
- **Closed form for any 2-state chain** A = [[1−a, a], [b, 1−b]]:
  Σ_{h=1..n} A^h = n·1π + c_n (I − 1π), with π = (b,a)/(a+b), λ = 1−a−b, c_n = λ(1−λⁿ)/(1−λ).
  This is unit-tested against explicit matrix powers.
- **Reduced mode:** the device's own reduced-mode delivery rate × (nominal interval / mode interval).

**Supervised models** (`hsm/m1/models.py`):
- Gradient-boosted trees (HistGradientBoosting, squared error, fixed seed).
- Features (`hsm/m1/features.py`, all causal):
  - recent and chronic service, reception, valid and mode availability;
  - current silence;
  - reception vs storage (backend signature);
  - outage history known so far;
  - fleet context: share of the family's other devices currently silent;
  - network-level flag;
  - calendar;
  - the individual model states and forecasts.
- Family-specific features:
  - radio aggregates (only for families that have radio metrics);
  - channel completeness (Dragino, Agvolution);
  - Sensoterra moisture-zero share (manufacturer contact signature);
  - CLOVER censored-GWC share.

## 3. Family-specific architecture
| level | candidates | role |
|---|---|---|
| global | `global_rate`, `gbm_global` (no family identity) | universal baselines |
| family | `family_rate`, `gbm_family` (only if ≥ 8 devices and ≥ 100 degraded episodes in the training data) | specialised supervised |
| individual | `persistence`, `chronic28`, `bernoulli`, `markov`, `hmm2` | per-sensor statistics, no supervised training |
| hierarchical | `eb_markov` | sensor shrunk toward its family |
| shared | `gbm_shared` (all families, family identity + descriptors, weights ∝ 1/√n_family) | information sharing without losing identity |
| hybrid | `hmm2_famcal` (individual HMM2, isotonic-calibrated per family) | individual dynamics + family calibration |

**Family profile** (`config.M1_*`):
- horizons per family: at least 4 slots, at most 3 days;
- primary horizon: 24 h, or 72 h for Agvolution and Bigbelly;
- data-sufficiency rule for supervised family models;
- family threshold τ_f,H = 10% quantile of realized availability in *normal* training windows (no declared outage, nominal mode), re-estimated from past data only.

**Validation:**
- Monthly walk-forward folds. Training only uses origins whose outcome window closed before the fold start (purged).
- Each choice uses only outcomes closed when it takes effect (`models.choose`):
  - August: pre-registered default (EB-Markov);
  - September: choice from August;
  - October–November (test): choice from August–September.
- Test outcomes are never used for any choice. Confidence intervals: device-block bootstrap.

## 4. States (`hsm/m1/state.py`)
Priority order:

1. **insufficient_information:** final partial hour of the data export, never heard, or < 24 h history.
2. **backend_data_unavailable:** data overdue while the device is still heard.
3. **significant_interruption:** causally declared outage, with its scope at declaration.
4. **reduced_reporting_mode:** reception-inferred, with its documented meaning.
5. **degraded:** forecast below τ_f, or a developing silence with S(k) < 0.01.
6. **normal.**

Traffic light: green / yellow / red / grey. Reasons and evidence text come with every state.

## 5. Results (final corrected foundation)
### Test-period comparison (Oct 1 to Nov 17, held out)

| family          |   H | selected    | selected_level   | selection_basis    |   n_test |   n_devices |   rmse_selected |   rmse_gbm_global | Δ vs gbm_global [95% CI]   |   rmse_universal | Δ vs universal [95% CI]    |
|:----------------|----:|:------------|:-----------------|:-------------------|---------:|------------:|----------------:|------------------:|:---------------------------|-----------------:|:---------------------------|
| Agvolution      |  24 | gbm_shared  | shared           | validation         |       16 |           3 |          0.1376 |            0.1838 | -0.0462 [-0.0790, -0.0342] |           0.1376 | +0.0000 [+0.0000, +0.0000] |
| Agvolution      |  72 | eb_markov   | hierarchical     | validation         |       16 |           3 |          0.5261 |            0.5397 | -0.0137 [-0.0807, +0.0344] |           0.5261 | +0.0000 [+0.0000, +0.0000] |
| Bigbelly        |  24 | markov      | individual       | validation         |     4460 |           4 |          0.134  |            0.1718 | -0.0378 [-0.0378, -0.0378] |           0.1757 | -0.0417 [-0.0417, -0.0417] |
| Bigbelly        |  72 | family_rate | family           | validation         |     4268 |           4 |          0.0781 |            0.1571 | -0.0789 [-0.0789, -0.0789] |           0.0742 | +0.0039 [+0.0039, +0.0039] |
| Dragino-SE0X-LB |   6 | gbm_shared  | shared           | universal_fallback |      802 |           1 |          0.0503 |            0.0503 | +0.0000 [+0.0000, +0.0000] |           0.0503 | +0.0000 [+0.0000, +0.0000] |
| Dragino-SE0X-LB |  24 | gbm_shared  | shared           | universal_fallback |      784 |           1 |          0.0324 |            0.0319 | +0.0005 [+0.0005, +0.0005] |           0.0324 | +0.0000 [+0.0000, +0.0000] |
| MeteoHelix      |   1 | hmm2        | individual       | validation         |    35230 |          32 |          0.1428 |            0.1353 | +0.0075 [+0.0035, +0.0120] |           0.1428 | +0.0000 [+0.0000, +0.0000] |
| MeteoHelix      |   6 | hmm2        | individual       | validation         |    35075 |          32 |          0.0858 |            0.0858 | +0.0000 [-0.0131, +0.0118] |           0.0827 | +0.0031 [-0.0085, +0.0137] |
| MeteoHelix      |  24 | persistence | individual       | validation         |    34517 |          32 |          0.0583 |            0.06   | -0.0017 [-0.0056, +0.0014] |           0.0626 | -0.0043 [-0.0078, -0.0012] |
| MeteoWind       |   1 | hmm2        | individual       | validation         |    33577 |          30 |          0.162  |            0.1594 | +0.0026 [+0.0011, +0.0043] |           0.162  | +0.0000 [+0.0000, +0.0000] |
| MeteoWind       |   6 | gbm_shared  | shared           | validation         |    33427 |          30 |          0.0905 |            0.0931 | -0.0026 [-0.0038, -0.0016] |           0.0905 | +0.0000 [+0.0000, +0.0000] |
| MeteoWind       |  24 | persistence | individual       | validation         |    32887 |          30 |          0.0626 |            0.0559 | +0.0067 [+0.0040, +0.0091] |           0.0565 | +0.0062 [+0.0032, +0.0087] |
| OttSensor       |  24 | hmm2        | individual       | validation         |     6690 |           6 |          0.0188 |            0.0219 | -0.0031 [-0.0043, -0.0030] |           0.0217 | -0.0029 [-0.0040, -0.0026] |
| Sensoterra      |   6 | gbm_shared  | shared           | validation         |    61298 |          62 |          0.2057 |            0.2051 | +0.0006 [+0.0003, +0.0008] |           0.2057 | +0.0000 [+0.0000, +0.0000] |
| Sensoterra      |  24 | gbm_shared  | shared           | validation         |    61206 |          62 |          0.1539 |            0.153  | +0.0009 [-0.0003, +0.0027] |           0.1539 | +0.0000 [+0.0000, +0.0000] |

### Pooled test RMSE by strategy

| strategy                                     |   RMSE origin-weighted |   RMSE family x horizon equal weight |
|:---------------------------------------------|-----------------------:|-------------------------------------:|
| per-family selection (specialised)           |                 0.1378 |                               0.175  |
| universal choice (one candidate per horizon) |                 0.1381 |                               0.1772 |
| gbm_global everywhere                        |                 0.1378 |                               0.1853 |
| gbm_shared everywhere                        |                 0.1379 |                               0.1784 |
| hmm2 everywhere                              |                 0.1451 |                               0.1788 |
| eb_markov everywhere                         |                 0.1496 |                               0.2168 |
| persistence everywhere                       |                 0.1684 |                               0.132  |

### Validation RMSE, all candidates

| family           |   H |   global_rate |   gbm_global |   family_rate |   gbm_family |   persistence |   chronic28 |   bernoulli |   markov |   hmm2 |   eb_markov |   gbm_shared |   hmm2_famcal |
|:-----------------|----:|--------------:|-------------:|--------------:|-------------:|--------------:|------------:|------------:|---------:|-------:|------------:|-------------:|--------------:|
| Agvolution       |  24 |         0.313 |        0.252 |         0.278 |      nan     |         0.299 |       0.271 |       0.271 |    0.266 |  0.264 |       0.265 |        0.251 |         0.265 |
| Agvolution       |  72 |         0.257 |        0.225 |         0.22  |      nan     |         0.248 |       0.217 |       0.217 |    0.215 |  0.222 |       0.213 |        0.223 |         0.213 |
| Bigbelly         |  24 |         0.274 |        0.247 |         0.195 |      nan     |         0.238 |       0.195 |       0.195 |    0.193 |  0.194 |       0.194 |        0.231 |         0.194 |
| Bigbelly         |  72 |         0.249 |        0.207 |         0.128 |      nan     |         0.196 |       0.132 |       0.132 |    0.132 |  0.132 |       0.132 |        0.204 |         0.129 |
| CloverSoilSensor |   1 |         0.468 |        0.246 |         0.47  |        0.249 |         0.307 |       0.27  |       0.311 |    0.311 |  0.286 |       0.311 |        0.247 |         0.274 |
| CloverSoilSensor |   6 |         0.335 |        0.147 |         0.425 |        0.16  |         0.166 |       0.18  |       0.239 |    0.241 |  0.221 |       0.241 |        0.145 |         0.201 |
| CloverSoilSensor |  24 |         0.39  |        0.126 |         0.419 |      nan     |         0.121 |       0.155 |       0.224 |    0.226 |  0.222 |       0.227 |        0.125 |         0.219 |
| MeteoHelix       |   1 |         0.249 |        0.187 |         0.233 |        0.149 |         0.189 |       0.152 |       0.152 |    0.152 |  0.147 |       0.152 |        0.186 |         0.152 |
| MeteoHelix       |   6 |         0.281 |        0.092 |         0.228 |      nan     |         0.094 |       0.091 |       0.091 |    0.09  |  0.084 |       0.09  |        0.087 |         0.089 |
| MeteoHelix       |  24 |         0.235 |        0.08  |         0.197 |      nan     |         0.058 |       0.069 |       0.069 |    0.069 |  0.066 |       0.069 |        0.068 |         0.074 |
| MeteoWind        |   1 |         0.254 |        0.193 |         0.251 |        0.175 |         0.214 |       0.177 |       0.177 |    0.177 |  0.167 |       0.177 |        0.193 |         0.173 |
| MeteoWind        |   6 |         0.256 |        0.096 |         0.229 |        0.106 |         0.108 |       0.11  |       0.11  |    0.11  |  0.097 |       0.11  |        0.094 |         0.102 |
| MeteoWind        |  24 |         0.217 |        0.077 |         0.202 |      nan     |         0.071 |       0.087 |       0.087 |    0.087 |  0.079 |       0.087 |        0.072 |         0.086 |
| OttSensor        |  24 |         0.314 |        0.03  |         0.03  |      nan     |         0.031 |       0.023 |       0.023 |    0.023 |  0.022 |       0.022 |        0.03  |         0.023 |
| Sensoterra       |   6 |         0.282 |        0.205 |         0.277 |        0.209 |         0.264 |       0.244 |       0.223 |    0.222 |  0.222 |       0.224 |        0.205 |         0.223 |
| Sensoterra       |  24 |         0.25  |        0.14  |         0.223 |        0.149 |         0.166 |       0.179 |       0.151 |    0.151 |  0.155 |       0.153 |        0.14  |         0.158 |

### Test RMSE, all candidates

| family          |   H |   global_rate |   gbm_global |   family_rate |   gbm_family |   persistence |   chronic28 |   bernoulli |   markov |   hmm2 |   eb_markov |   gbm_shared |   hmm2_famcal |
|:----------------|----:|--------------:|-------------:|--------------:|-------------:|--------------:|------------:|------------:|---------:|-------:|------------:|-------------:|--------------:|
| Agvolution      |  24 |         0.699 |        0.184 |         0.541 |      nan     |         0     |       0.544 |       0.544 |    0.472 |  0.335 |       0.491 |        0.138 |         0.302 |
| Agvolution      |  72 |         0.672 |        0.54  |         0.544 |      nan     |         0.07  |       0.544 |       0.544 |    0.515 |  0.439 |       0.526 |        0.517 |         0.484 |
| Bigbelly        |  24 |         0.255 |        0.172 |         0.135 |      nan     |         0.19  |       0.132 |       0.133 |    0.134 |  0.132 |       0.132 |        0.176 |         0.131 |
| Bigbelly        |  72 |         0.256 |        0.157 |         0.078 |      nan     |         0.116 |       0.073 |       0.075 |    0.077 |  0.074 |       0.074 |        0.158 |         0.074 |
| Dragino-SE0X-LB |   6 |         0.291 |        0.05  |         0.219 |      nan     |         0.068 |       0.17  |       0.048 |    0.049 |  0.049 |       0.059 |        0.05  |         0.05  |
| Dragino-SE0X-LB |  24 |         0.256 |        0.032 |         0.193 |      nan     |         0.032 |       0.167 |       0.025 |    0.025 |  0.026 |       0.043 |        0.032 |         0.027 |
| MeteoHelix      |   1 |         0.227 |        0.135 |         0.207 |        0.138 |         0.18  |       0.148 |       0.15  |    0.149 |  0.143 |       0.149 |        0.135 |         0.143 |
| MeteoHelix      |   6 |         0.253 |        0.086 |         0.171 |        0.11  |         0.084 |       0.091 |       0.094 |    0.094 |  0.086 |       0.094 |        0.083 |         0.087 |
| MeteoHelix      |  24 |         0.224 |        0.06  |         0.165 |      nan     |         0.058 |       0.078 |       0.081 |    0.081 |  0.077 |       0.081 |        0.063 |         0.079 |
| MeteoWind       |   1 |         0.225 |        0.159 |         0.221 |        0.16  |         0.21  |       0.168 |       0.168 |    0.167 |  0.162 |       0.167 |        0.159 |         0.163 |
| MeteoWind       |   6 |         0.222 |        0.093 |         0.172 |        0.109 |         0.103 |       0.095 |       0.095 |    0.095 |  0.09  |       0.095 |        0.09  |         0.091 |
| MeteoWind       |  24 |         0.191 |        0.056 |         0.159 |      nan     |         0.063 |       0.069 |       0.069 |    0.069 |  0.067 |       0.069 |        0.056 |         0.068 |
| OttSensor       |  24 |         0.289 |        0.022 |         0.024 |      nan     |         0.027 |       0.018 |       0.018 |    0.018 |  0.019 |       0.019 |        0.022 |         0.019 |
| Sensoterra      |   6 |         0.276 |        0.205 |         0.266 |        0.208 |         0.261 |       0.233 |       0.227 |    0.226 |  0.221 |       0.226 |        0.206 |         0.221 |
| Sensoterra      |  24 |         0.245 |        0.153 |         0.215 |        0.16  |         0.176 |       0.173 |       0.168 |    0.167 |  0.165 |       0.168 |        0.154 |         0.164 |

### Test: detection of "availability below family threshold" by the selected model

| family          |   H | candidate   |     n |   event_rate |     auc |      ap |   precision |   recall |   bias |
|:----------------|----:|:------------|------:|-------------:|--------:|--------:|------------:|---------:|-------:|
| Agvolution      |  24 | gbm_shared  |    16 |        1     | nan     | nan     |       1     |    0.625 |  0.122 |
| Agvolution      |  72 | eb_markov   |    16 |        1     | nan     | nan     |     nan     |    0     |  0.522 |
| Bigbelly        |  24 | markov      |  4460 |        0.02  |   0.529 |   0.023 |     nan     |    0     | -0.029 |
| Bigbelly        |  72 | family_rate |  4268 |        0     | nan     | nan     |     nan     |  nan     | -0.041 |
| Dragino-SE0X-LB |   6 | gbm_shared  |   802 |        0.121 |   0.597 |   0.179 |       0.191 |    0.268 |  0     |
| Dragino-SE0X-LB |  24 | gbm_shared  |   784 |        0.485 |   0.852 |   0.829 |       0.815 |    0.913 | -0.006 |
| MeteoHelix      |   1 | hmm2        | 35230 |        0.054 |   0.933 |   0.489 |       0.558 |    0.535 | -0.002 |
| MeteoHelix      |   6 | hmm2        | 35075 |        0.069 |   0.964 |   0.789 |       0.84  |    0.685 | -0.002 |
| MeteoHelix      |  24 | persistence | 34517 |        0.07  |   0.99  |   0.893 |       0.844 |    0.839 | -0     |
| MeteoWind       |   1 | hmm2        | 33577 |        0.039 |   0.914 |   0.436 |       0.595 |    0.425 | -0     |
| MeteoWind       |   6 | gbm_shared  | 33427 |        0.052 |   0.978 |   0.838 |       0.901 |    0.704 | -0.02  |
| MeteoWind       |  24 | persistence | 32887 |        0.047 |   0.988 |   0.886 |       0.804 |    0.817 | -0.001 |
| OttSensor       |  24 | hmm2        |  6690 |        0.053 |   0.925 |   0.274 |       0.295 |    0.794 | -0.003 |
| Sensoterra      |   6 | gbm_shared  | 61298 |        0.106 |   0.863 |   0.597 |       0.901 |    0.272 |  0.021 |
| Sensoterra      |  24 | gbm_shared  | 61206 |        0.133 |   0.839 |   0.63  |       0.822 |    0.384 |  0.014 |

### Data sufficiency and family thresholds at the test start

| family          |   H |   train_devices |   train_degraded_episodes | supervised_family_model   |   tau |
|:----------------|----:|----------------:|--------------------------:|:--------------------------|------:|
| MeteoHelix      |   1 |              32 |                      2539 | True                      | 0.5   |
| MeteoWind       |   1 |              30 |                      1518 | True                      | 0.333 |
| MeteoHelix      |   6 |              32 |                       201 | True                      | 0.556 |
| MeteoWind       |   6 |              30 |                       284 | True                      | 0.472 |
| Dragino-SE0X-LB |   6 |               0 |                         0 | False                     | 0.944 |
| Sensoterra      |   6 |              63 |                      3184 | True                      | 0.2   |
| MeteoHelix      |  24 |              32 |                        26 | False                     | 0.569 |
| MeteoWind       |  24 |              30 |                        59 | False                     | 0.507 |
| Dragino-SE0X-LB |  24 |               0 |                         0 | False                     | 0.986 |
| Sensoterra      |  24 |              63 |                       825 | True                      | 0.375 |
| OttSensor       |  24 |               6 |                        39 | False                     | 0.958 |
| Agvolution      |  24 |               8 |                        79 | False                     | 0.167 |
| Bigbelly        |  24 |               4 |                        48 | False                     | 0.6   |
| Agvolution      |  72 |               8 |                        48 | False                     | 0.278 |
| Bigbelly        |  72 |               4 |                        24 | False                     | 0.75  |

### Most informative information groups (permutation, test origins)

| family          |   H | model      |   base_rmse | top_information                                                                                                                          |
|:----------------|----:|:-----------|------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|
| Bigbelly        |  72 | gbm_shared |       0.191 | individual_state_models (+0.0037), recent_service_availability (+0.0023), chronic_availability (+0.0022), calendar (+0.0007)             |
| Dragino-SE0X-LB |  24 | gbm_shared |       0.032 | current_silence (+0.0005), individual_forecasts (+0.0000), family_identity (+0.0000), family_specific_measurement (+0.0000)              |
| MeteoHelix      |   6 | gbm_family |       0.133 | recent_service_availability (+0.0623), radio (+0.0072), individual_state_models (+0.0058), outage_history (+0.0049)                      |
| MeteoHelix      |   6 | gbm_shared |       0.096 | recent_service_availability (+0.0519), reception_vs_storage (+0.0168), current_silence (+0.0019), radio (+0.0015)                        |
| MeteoHelix      |  24 | gbm_shared |       0.069 | recent_service_availability (+0.0277), chronic_availability (+0.0217), reception_vs_storage (+0.0213), individual_state_models (+0.0057) |
| MeteoWind       |   6 | gbm_family |       0.126 | recent_service_availability (+0.0425), chronic_availability (+0.0085), calendar (+0.0020), outage_history (+0.0005)                      |
| MeteoWind       |   6 | gbm_shared |       0.101 | recent_service_availability (+0.0462), reception_vs_storage (+0.0131), calendar (+0.0016), data_validity_and_mode (+0.0016)              |
| MeteoWind       |  24 | gbm_shared |       0.059 | chronic_availability (+0.0245), recent_service_availability (+0.0231), reception_vs_storage (+0.0209), individual_state_models (+0.0077) |
| OttSensor       |  24 | gbm_shared |       0.022 | individual_state_models (+0.0070), recent_service_availability (+0.0066), reception_vs_storage (+0.0039), chronic_availability (+0.0009) |
| Sensoterra      |   6 | gbm_family |       0.21  | recent_service_availability (+0.0212), current_silence (+0.0089), radio (+0.0054), data_validity_and_mode (+0.0038)                      |
| Sensoterra      |   6 | gbm_shared |       0.206 | recent_service_availability (+0.0318), reception_vs_storage (+0.0056), current_silence (+0.0047), radio (+0.0040)                        |
| Sensoterra      |  24 | gbm_family |       0.16  | recent_service_availability (+0.0159), individual_forecasts (+0.0054), radio (+0.0047), chronic_availability (+0.0041)                   |
| Sensoterra      |  24 | gbm_shared |       0.155 | reception_vs_storage (+0.0163), recent_service_availability (+0.0137), chronic_availability (+0.0131), current_silence (+0.0035)         |

### Share of device-hours per state (%)

| family           |   backend_data_unavailable |   degraded |   insufficient_information |   normal |   reduced_reporting_mode |   significant_interruption |
|:-----------------|---------------------------:|-----------:|---------------------------:|---------:|-------------------------:|---------------------------:|
| Adeunis          |                        0   |        3.8 |                        0.8 |      0   |                      0   |                       95.5 |
| Agvolution       |                       10.6 |       20   |                        0.8 |     44.9 |                      0   |                       23.7 |
| Bigbelly         |                        0   |       24.8 |                        0.7 |     74.6 |                      0   |                        0   |
| CloverSoilSensor |                        4.5 |       13.9 |                        0.7 |     33.7 |                      0   |                       47.2 |
| Dragino-SE0X-LB  |                        8.3 |       54.6 |                        4.7 |     32.5 |                      0   |                        0   |
| MeteoHelix       |                        0   |       28.5 |                        8   |     62.6 |                      0   |                        0.8 |
| MeteoWind        |                        0   |       30.2 |                        1   |     68.6 |                      0   |                        0.2 |
| OttSensor        |                        0   |       11.1 |                        0.4 |     88.4 |                      0   |                        0   |
| Sensoterra       |                        0.4 |       12.4 |                        0.7 |     68.7 |                      4.5 |                       13.3 |

### Current state per family (last complete hour)

| family           |   backend_data_unavailable |   degraded |   normal |   significant_interruption |
|:-----------------|---------------------------:|-----------:|---------:|---------------------------:|
| Adeunis          |                          0 |          0 |        0 |                          1 |
| Agvolution       |                          1 |          0 |        0 |                          8 |
| Bigbelly         |                          0 |          0 |        4 |                          0 |
| CloverSoilSensor |                          1 |          0 |        0 |                          9 |
| Dragino-SE0X-LB  |                          0 |          0 |        1 |                          0 |
| MeteoHelix       |                          0 |          2 |       29 |                          1 |
| MeteoWind        |                          0 |          2 |       27 |                          1 |
| OttSensor        |                          0 |          0 |        6 |                          0 |
| Sensoterra       |                         20 |          0 |        2 |                         42 |

### Lifetime availability (median device) and loss behaviour

| family           |   service_availability |   reception_availability |   valid_data_availability |   mode_availability |   share_cycles_all_channels |   p_miss_28d |   p_miss_after_miss_28d |   burstiness_ratio |
|:-----------------|-----------------------:|-------------------------:|--------------------------:|--------------------:|----------------------------:|-------------:|------------------------:|-------------------:|
| Adeunis          |                  0.004 |                    0.004 |                     0.004 |               0.004 |                       1     |      nan     |                 nan     |            nan     |
| Agvolution       |                  0.42  |                    0.412 |                     0.112 |               0.42  |                       0.26  |      nan     |                 nan     |            nan     |
| Bigbelly         |                  0.883 |                    0.883 |                     0.883 |               0.883 |                       1     |        0.071 |                   0     |              0     |
| CloverSoilSensor |                  0.323 |                    0.324 |                     0.318 |               0.323 |                       0.999 |      nan     |                 nan     |            nan     |
| Dragino-SE0X-LB  |                  0.858 |                    0.961 |                     0.858 |               0.858 |                       1     |        0.039 |                   0.063 |              1.615 |
| MeteoHelix       |                  0.877 |                    0.869 |                     0.877 |               0.877 |                       1     |        0.082 |                   0.134 |              1.168 |
| MeteoWind        |                  0.79  |                    0.79  |                     0.79  |               0.79  |                       1     |        0.187 |                   0.218 |              1.171 |
| OttSensor        |                  0.999 |                    0.999 |                     0.999 |               0.999 |                       1     |        0     |                   0.19  |              4.395 |
| Sensoterra       |                  0.566 |                    0.574 |                     0.566 |               0.583 |                       1     |        0.365 |                   0.396 |              1.055 |


## 6. Interpretation
- **Specialisation helps where families differ most.** With every family × horizon weighted equally:
  - per-family selection: RMSE 0.1750;
  - universal choice: 0.1772;
  - `gbm_shared` everywhere: 0.1784;
  - `gbm_global` everywhere: 0.1853.
  - Origin-weighted the strategies tie (≈ 0.138), because the large families select shared models anyway.
- **Individual statistical models win for small, regular families.**
  - OttSensor: hmm2 0.0188 vs gbm_global 0.0219, CI excludes 0.
  - Bigbelly 24 h: markov 0.134 vs 0.172.
  - Dragino: markov/bernoulli 0.025 on test vs selected `gbm_shared` 0.032. Dragino had no validation data, so the universal fallback was used; a better pre-registered default for families without validation is the hierarchical/individual model.
- **Shared learning beats separate family models.**
  - `gbm_family` was trained wherever data sufficed, but never won validation: 0.209 vs 0.205 (Sensoterra 6 h), 0.110 vs 0.083 (MeteoHelix 6 h, test).
  - Pooling families with family identity is better than isolating them.
- **Selection noise is visible.**
  - MeteoWind 24 h chose persistence on validation (0.071 vs 0.072) but is worse on test (0.063 vs 0.056).
  - For the 10-min families at 1 h, HMM2 won validation, but the shared model improved as training data grew (test 0.135 vs 0.143).
  - The pre-registered results are reported unchanged; a stability rule decided on future validation data is a candidate refinement.
- **Detection of availability below the family threshold** (test, selected model):
  - MeteoHelix/MeteoWind: AUC 0.91–0.99, AP 0.44–0.89.
  - Sensoterra: AUC 0.84–0.86, precision 0.82–0.90, recall 0.27–0.38.
  - Ott: AUC 0.93.
  - Short-horizon (1 h) detection is hardest. Losses are mostly random at that scale (burstiness ≈ 1.1–1.2 for the weather stations).
- **What matters (permutation importance):**
  - recent service availability dominates everywhere;
  - reception-vs-storage information is the second group for the weather stations and Sensoterra;
  - chronic availability matters at 24 h;
  - individual state models matter most for Ott and Bigbelly;
  - radio adds a little for Sensoterra and MeteoHelix.

## 7. Verified findings from the state layer
- **Backend state validated independently.** The 20 Sensoterra probes in `backend_data_unavailable` at the end of the data are exactly the 20 probes whose inserts fail with ProgrammingError since 13 Nov 12:55. The state rule only uses reception-vs-storage timing.
- **Agvolution delivers two separate messages per cycle.** In 74% of cycles one of them is lost: some data in 42% of expected cycles, complete data in 11%.
- **Dragino:** 11% of uplinks carry only metadata rows. Service 0.858 vs reception 0.961.
- **Bigbelly:** the 4 bins have identical delivery outcomes. They are effectively one independent unit, so device-level confidence intervals are degenerate.
- **CLOVER:** the whole fleet has been in a declared outage since 1 Sep (fleet removal), so there are no test origins. Validation (August) only.

## 8. Limitations
- **Test periods are short:** 1.5 months, and Agvolution has only 16 scorable test origins.
- **No validation data** for Dragino, so its choice is a fallback.
- **Forecasts exclude declared outages at the origin.** Persistence of an ongoing outage is Module 3.
- **The HMM2 is fitted without outage periods**, so it underestimates outage-onset risk. The family calibration (`hmm2_famcal`) and the supervised models absorb part of this.
- **Permutation importance** of correlated groups is indicative, not causal.
- **Thresholds and horizons** are operational definitions with stated provenance.

## 9. Outputs for subsequent modules
| table | content | used by |
|---|---|---|
| `gold_m1_features` | causal per-sensor-hour features, individual model states (HMM2 P(bad), silence surprise, EB/Markov parameters) | M2, M3, M7, M8 |
| `gold_m1_predictions` | walk-forward forecasts of every candidate, per horizon | M7, M10, M11 |
| `gold_m1_state` | state, traffic light, reasons, forecast, family threshold, expected missing reports | M10, M12 |
| `gold_m1_period_daily/weekly` | packet loss and the four availabilities per period, missed reports, longest silence, backend counts, channel completeness | M8, M12 (challenge: loss per sensor and period) |
| `gold_m1_device_summary` | lifetime statistics, loss burstiness, outage history by scope and cause, state shares, current state, evidence | M10, M12 |
| `gold_m1_scores/comparison/fold_info/importance` | evaluation | report, M11 |
