# PPG Signal Processing — Background, Plotting Guide & Roadmap
**AS7058 Multi-Site Dataset | March 2026**

---

## 1. What Is PPG? (The Physics in 2 Minutes)

**Photoplethysmography (PPG)** works by shining light into tissue and measuring how much comes back.

```
LED (light source)
      ↓  light into tissue
   [skin / blood vessels]
      ↑  reflected light
Photodiode (detector)
      ↓
AS7058 ADC → raw number (e.g. 337,450)
```

When your heart beats, more blood is pushed into the capillaries. That extra blood absorbs more light,
so **less light returns to the sensor**. This creates a small rhythmic dip in the signal — one dip per heartbeat.

### Two Light Wavelengths Used

| Channel    | Wavelength | AS7058 Column | Why Used |
|-----------|-----------|--------------|----------|
| **IR**     | ~850 nm   | PPG1_SUB1    | Good for HR detection, less affected by skin pigment |
| **RED**    | ~660 nm   | PPG1_SUB2    | Essential for SpO2 — oxygenated vs deoxygenated blood absorb differently |
| **3rd**    | ~green    | PPG1_SUB3    | Ambient/green reference (motion artifact cancellation) |

---

## 2. Anatomy of a Raw PPG Signal

```
ADC value
  |
  |  337,500 ──────────────────────────── DC baseline (mean tissue absorption)
  |              ↑           ↑
  |  337,460   _/ \_       _/ \_         AC component = pulsatile (heartbeat)
  |  337,420  /     \_   _/     \_       peak-to-peak ~ 40-100 ADC counts
  |          /         \_/
  |
  └────────────────────────────────────── time (seconds)
         ↑ diastole (less blood)    ↑ systole (more blood returns, signal dips)
```

### Two Components — Always Remember This

| Component | Name | Value (typical) | What it tells you |
|-----------|------|-----------------|-------------------|
| **DC**    | Baseline offset | ~337,000–900,000 ADC | Tissue thickness, LED power, skin colour |
| **AC**    | Pulsatile ripple | ~100–10,000 ADC | Actual heartbeat signal |

The **AC/DC ratio** is called **Perfusion Index (PI%)** and is the single most important quality metric:

```
PI (%) = (AC peak-to-peak / DC mean) × 100

Finger:  PI typically 1–5%   → GOOD
Wrist:   PI typically 0.3–2% → ACCEPTABLE
Chest:   PI typically 0.05%  → POOR (with wrist algorithm)
```

---

## 3. Key Signal Processing Concepts

### 3.1 Sampling Rate (Fs)

The AS7058 captures a new sample every fixed time interval.

```
Our data:  Wrist/Chest = 200 Hz (200 samples per second)
           Finger      = 100 Hz (100 samples per second)

Nyquist rule: You can detect signals up to Fs/2
  → 100 Hz sensor can detect up to 50 Hz (heart rate at 3,000 BPM — way more than needed)
  → Heartbeat is ~1 Hz (60 BPM), so even 10 Hz would technically be enough for HR
```

### 3.2 Filtering

Raw signal contains everything: heartbeat, breathing, motion, electrical noise.
Filters keep only the frequencies you care about.

```
Raw PPG signal frequencies:
  0 Hz         → DC offset (very large, useless number like 337,000)
  0.1–0.5 Hz   → Breathing (respiratory rate, 6–30 breaths/min)
  0.7–3.5 Hz   → Heartbeat (42–210 BPM)
  > 4 Hz       → Noise (motion artifact, electrical interference)

Filter types used:
  Highpass  (cutoff 0.05 Hz)  → removes DC baseline, reveals AC waveform
  Bandpass  (0.5–4 Hz)        → keeps ONLY heartbeat band
  Lowpass   (cutoff 0.5 Hz)   → keeps only breathing and below
```

**Analogy:** Think of filtering like an equaliser on a speaker. You boost the frequency range you want and cut everything else.

### 3.3 FFT / Power Spectral Density (PSD)

FFT breaks your signal into its frequency components — like a musical chord into individual notes.

```
Time domain:                   Frequency domain (PSD):
signal amplitude               power
    │ ╭╮  ╭╮  ╭╮                  │
    │╭╯╰╮╭╯╰╮╭╯╰╮                 │    ▌
    │╯  ╰╯  ╰╯  ╰──               │    █
    └───────────── time           └────█──────── frequency
                                       ↑
                                    HR peak
                                    ~1 Hz = 60 BPM
```

**SNR (Signal-to-Noise Ratio)** from PSD:
```
SNR (dB) = 10 × log10(power in HR band / power in noise band)

> 6 dB  → heartbeat peak is clearly visible, algorithm will work
3–6 dB  → marginal, algorithm may struggle
< 3 dB  → noise dominant, algorithm will fail
```

### 3.4 AGC — Automatic Gain Control

The AS7058 automatically adjusts LED brightness to get a good signal back.

```
Low LED current (e.g. 15 LSB)  → sensor got plenty of light back → GOOD contact
High LED current (e.g. 160 LSB) → sensor is at maximum brightness, still weak → POOR contact

Our results:
  Finger V3: AGC settled at  15 LSB  ← excellent optical coupling
  Wrist V1:  AGC settled at  80 LSB  ← moderate
  Chest:     AGC stuck at   160 LSB  ← sensor at max, still not enough signal
```

---

## 4. How to Plot PPG — Step by Step

### Step 1: Plot the Raw Signal First
```python
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv('your_file.csv', low_memory=False)
df['ir']  = pd.to_numeric(df['PPG1_SUB1'], errors='coerce')
df['red'] = pd.to_numeric(df['PPG1_SUB2'], errors='coerce')
df['t']   = pd.to_numeric(df['TIMESTAMP [s]'], errors='coerce')

ppg = df[df['ir'].notna()].sort_values('t')

plt.figure(figsize=(14, 4))
plt.plot(ppg['t'], ppg['ir'],  label='IR  (SUB1)', color='blue', lw=0.5)
plt.plot(ppg['t'], ppg['red'], label='RED (SUB2)', color='red',  lw=0.5)
plt.xlabel('Time (seconds)')
plt.ylabel('ADC Counts (raw, includes large DC offset)')
plt.title('Raw PPG — both channels')
plt.legend()
plt.show()
```

**What you will see:** Two nearly flat lines at ~337,000 (IR) and ~600,000 (RED).
The heartbeat is invisible because the DC offset (337,000) dwarfs the AC signal (~100 counts).

---

### Step 2: Remove the DC Baseline (Highpass Filter)
```python
from scipy.signal import butter, filtfilt

def highpass(sig, fs, cutoff=0.05):
    b, a = butter(2, cutoff / (fs/2), btype='high')
    return filtfilt(b, a, sig)

fs  = 100  # or 200, check your file
ir  = ppg['ir'].values.astype(float)
t   = ppg['t'].values

ir_ac = highpass(ir, fs, cutoff=0.05)   # removes DC, keeps everything > 0.05 Hz

plt.figure(figsize=(14, 4))
plt.plot(t, ir_ac, color='blue', lw=0.7, label='IR (DC removed)')
plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.xlabel('Time (seconds)')
plt.ylabel('Delta ADC Counts (AC component)')
plt.title('IR PPG after DC removal — now you can see the heartbeat waveform')
plt.legend()
plt.show()
```

**What you will now see:** Clear up-down waves, one per heartbeat.

---

### Step 3: Bandpass Filter to Clean the Signal
```python
from scipy.signal import butter, filtfilt

def bandpass(sig, fs, lo=0.5, hi=4.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo/nyq, min(hi/nyq, 0.99)], btype='band')
    return filtfilt(b, a, sig)

ir_bp = bandpass(ir, fs, lo=0.5, hi=4.0)   # keeps only heartbeat band

plt.figure(figsize=(14, 4))
plt.plot(t, ir_bp, color='blue', lw=0.8, label='IR bandpass (0.5–4 Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Delta ADC Counts')
plt.title('Bandpass filtered IR PPG — clean heartbeat signal')
plt.legend()
plt.show()
```

---

### Step 4: Detect Heartbeat Peaks
```python
from scipy.signal import find_peaks
import numpy as np

dist = int(fs * 0.4)               # minimum 0.4s between peaks (= max 150 BPM)
thr  = np.median(ir_bp) + 0.4 * np.std(ir_bp)
peaks, _ = find_peaks(ir_bp, distance=dist, height=thr)

hr_bpm = 60.0 / (np.median(np.diff(peaks)) / fs)   # median inter-peak interval → HR

plt.figure(figsize=(14, 4))
plt.plot(t, ir_bp, color='blue', lw=0.8, label='Filtered IR')
plt.plot(t[peaks], ir_bp[peaks], 'v', color='orange', ms=8, label=f'Peaks  HR≈{hr_bpm:.0f} BPM')
plt.xlabel('Time (seconds)')
plt.ylabel('Delta ADC Counts')
plt.title(f'Peak Detection — Estimated HR = {hr_bpm:.0f} BPM')
plt.legend()
plt.show()
```

---

### Step 5: Compute and Plot PSD (FFT-based)
```python
from scipy.signal import welch

f, psd = welch(ir_bp, fs=fs, nperseg=int(fs * 8))

plt.figure(figsize=(10, 4))
plt.semilogy(f, psd, color='blue', lw=1.2)
plt.axvspan(0.7, 3.5, alpha=0.15, color='green', label='HR band')
plt.axvspan(0.1, 0.5, alpha=0.10, color='purple', label='Breathing band')
plt.xlim(0, 5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (ADC² / Hz)  —  log scale')
plt.title('PSD — the tall spike in the green band = your heart rate')
plt.legend()
plt.show()
```

---

### Step 6: Compute Perfusion Index (PI%)
```python
# PI = AC amplitude / DC level × 100
dc    = np.mean(np.abs(red))          # DC: mean of raw RED signal
ac_pp = np.ptp(bandpass(red, fs))     # AC: peak-to-peak of bandpass filtered RED
pi    = ac_pp / dc * 100

print(f'Perfusion Index = {pi:.2f}%')
# > 1.0% = good,  0.3–1.0% = acceptable,  < 0.3% = poor
```

---

## 5. What Our Data Shows — Summary

### Signal Quality Ranking (from our measurements)

```
Rank | Dataset      | PI%   | AGC  | HR peaks | Verdict
-----|--------------|-------|------|----------|---------------------------
  1  | Finger V3    | 2.76  |  15  | Clear    | BEST — use for SpO2 calib.
  2  | Wrist V1     | 3.08  |  80  | OK       | GOOD for HR, noisy for SpO2
  3  | Finger V1    | 1.94  |  50  | Clear    | GOOD — reliable
  4  | Finger V2    | 1.64  |  50  | Clear    | GOOD — reliable
  5  | Chest V2     | 0.38  | 160  | Marginal | BORDERLINE — HR trend only
  6  | Wrist V2     | 0.16  | 160  | Poor     | POOR — contact issue
  7  | Chest V1     | 0.12  | 160  | Noise    | POOR — discard
```

### Key Finding: RED channel vs IR channel

```
IR  (SUB1): AC amplitude ~10–300 ADC counts   (small pulsatile component)
RED (SUB2): AC amplitude ~1,000–15,000 ADC    (10–50× larger pulsatile component)

→ Use RED channel for SpO2 and PI calculations
→ Use IR channel for HR peak detection (more stable baseline)
→ Both channels needed for R-ratio → SpO2 formula
```

### Key Finding: Sensor Firmware Status

The AS7058 firmware IS running its internal SpO2 algorithm and outputs at ~1 Hz:
- `SPO2: SIGNAL_QUALITY` — firmware confidence score
- `SPO2: PI [%]` — firmware-computed PI
- `SPO2: R` — R-ratio (direct SpO2 input)
- `SPO2: HEART_RATE [bpm]` — firmware HR estimate
- `SPO2: SPO2 [%]` — firmware SpO2 estimate

**However:** `AC_RED`, `DC_RED`, `AC_IR`, `DC_IR` columns are all zeros — firmware computes them internally but does NOT export the intermediate values. We must derive these ourselves from raw PPG.

---

## 6. The SpO2 Math (For Future Algorithm Work)

SpO2 is computed from the ratio of RED to IR pulsatile signals:

```
Step 1: Compute AC and DC for both channels
   DC_IR  = mean(IR_raw)
   AC_IR  = peak-to-peak of bandpass(IR_raw)
   DC_RED = mean(RED_raw)
   AC_RED = peak-to-peak of bandpass(RED_raw)

Step 2: Compute R-ratio
   R = (AC_RED / DC_RED) / (AC_IR / DC_IR)

Step 3: Apply empirical formula
   SpO2 (%) = 104 - 17 × R      ← calibration for fingertip transmission
   SpO2 (%) = 110 - 25 × R      ← alternative calibration (Beer-Lambert based)

Typical R values:
   R = 0.5  → SpO2 ≈ 96%  (normal healthy)
   R = 1.0  → SpO2 ≈ 87%
   R = 0.4  → SpO2 ≈ 97%

IMPORTANT: The a and b coefficients in (a - b×R) are device-specific.
You must calibrate them against the SP-20 reference data (Finger V3 dataset).
```

---

## 7. HRV, RR, and Advanced Metrics

Once you have clean peak detection working, these come directly from the RR intervals:

```python
# rr = array of time gaps between consecutive heartbeat peaks (in seconds)
rr = np.diff(peaks) / fs   # e.g. [0.82, 0.79, 0.81, 0.83, ...]

# Heart Rate (beat-by-beat)
hr_per_beat = 60.0 / rr           # BPM for each beat

# HRV — Time Domain
sdnn  = np.std(rr) * 1000         # ms — overall HRV, normal 30–100 ms
rmssd = np.sqrt(np.mean(np.diff(rr)**2)) * 1000   # ms — short-term HRV
pnn50 = np.mean(np.abs(np.diff(rr)) > 0.05) * 100 # % beats differing >50ms

# Respiratory Rate (from PPG envelope modulation)
envelope = np.abs(scipy.signal.hilbert(ir_bp))     # amplitude envelope
# The envelope oscillates at the breathing frequency (0.1–0.5 Hz)
# Find dominant frequency in that band → breathing rate in breaths/min
```

---

## 8. Future TODO — Development Roadmap

### Phase 1: Signal Quality (DONE ✓)
- [x] Load and parse all AS7058 CSV files (two schemas A and B)
- [x] Compute PI%, SNR, AC amplitude, DC baseline for all datasets
- [x] Generate quality heatmap, spectrogram, beat templates
- [x] Confirm Finger V3 as best dataset for calibration
- [x] Confirm Chest is currently not viable with wrist algorithm

### Phase 2: HR Algorithm (Next — 1–2 weeks)
- [ ] Peak detection on RED channel (higher amplitude, cleaner peaks)
- [ ] Ectopic beat rejection: remove RR intervals > ±20% of running median
- [ ] Validate derived HR against `SPO2: HEART_RATE [bpm]` (sensor ground truth)
- [ ] Validate against SP-20 HR (for Finger V3 parallel recording)
- [ ] Tune `find_peaks` parameters per body site (finger vs wrist need different thresholds)

### Phase 3: SpO2 Algorithm (2–3 weeks)
- [ ] Compute AC/DC per channel using sliding 4-second windows
- [ ] Compute R-ratio from raw PPG: R = (AC_RED/DC_RED) / (AC_IR/DC_IR)
- [ ] Compare computed R to sensor's `SPO2: R` column — should agree
- [ ] Calibrate empirical formula: plot sensor R vs SP-20 SpO2 → fit a, b in (a - b×R)
- [ ] Validate calibrated formula on Finger V1 and V2 (held-out data)

### Phase 4: HRV (3–4 weeks)
- [ ] Implement ectopic beat removal pipeline
- [ ] Compute SDNN, RMSSD, pNN50 from clean RR intervals
- [ ] Validate minimum recording length needed (usually 5 minutes for SDNN)
- [ ] Wrist data: add motion artifact rejection using ACC data (Wrist schema A)

### Phase 5: Respiratory Rate — RR (4–5 weeks)
- [ ] Extract PPG envelope using Hilbert transform
- [ ] Bandpass envelope at 0.1–0.5 Hz
- [ ] Validate on a recording where subject controls breathing rate
- [ ] Compare against known reference (no reference device currently available)

### Phase 6: Chest Algorithm Improvement
- [ ] Obtain MXREFDES106 reference platform (chest-optimised optical design)
- [ ] Re-test chest at different LED power / gain settings
- [ ] Apply chest-specific bandpass tuning (0.3–2.5 Hz vs wrist 0.5–4 Hz)
- [ ] Target PI > 0.3% for chest to be viable

### Phase 7: System Integration & Validation
- [ ] All algorithms running in real-time on firmware
- [ ] Clinical validation: 20+ subjects, compare to FDA-cleared reference
- [ ] ISO 80601-2-61 compliance testing for SpO2 accuracy (RMSE < 3.5%)

---

## 9. Files Created in This Project

```
02.03.2023/
├── ppg_analysis.py           Original full analysis script (9 plots)
├── ppg_quality_analysis.py   Per-dataset quality plots (12 plots A-F)
├── ppg_signal_quality.py     Detailed quality metrics (12 plots 01-12)
├── one_plot_comparison.py    Single comparison plot (Finger vs Wrist vs Chest)
│
├── output/                   Output from ppg_analysis.py
├── output_quality/           Output from ppg_quality_analysis.py
└── output_sq/                Output from ppg_signal_quality.py
    ├── 01_pi_over_time.png
    ├── 02_ac_amplitude_over_time.png
    ├── 03_dc_baseline_over_time.png
    ├── 04_snr_over_time.png
    ├── 05_spectrogram.png
    ├── 06_psd_per_dataset.png
    ├── 07_beat_template.png
    ├── 08_ir_red_correlation.png
    ├── 09_sensor_events.png
    ├── 10_agc_settling.png
    ├── 11_quality_heatmap.png
    ├── 12_waveform_overlay_30s.png
    └── SIGNAL_QUALITY_COMPARISON.png  ← The single best summary plot
```

---

## 10. Quick Reference — Axis Conventions

| Plot Type | X Axis | Y Axis |
|-----------|--------|--------|
| Raw PPG | Time (seconds) | ADC Counts (raw integer, large DC offset included) |
| AC-filtered PPG | Time (seconds) | Delta ADC Counts (baseline removed, centred at 0) |
| Perfusion Index | Time (seconds) | PI (%) — higher is better |
| PSD / FFT | Frequency (Hz) | Power (ADC²/Hz) — log scale, look for peak at HR frequency |
| SNR | Time (seconds) | SNR (dB) — above 6 dB is usable |
| Spectrogram | Time (seconds) | Frequency (Hz) — bright stripe = dominant frequency |
| Beat template | Time from peak (ms) | Normalised amplitude (−1 to +1) |
| Bland-Altman | Mean of two methods | Difference (method A − method B) |
| HR timeseries | Time (seconds) | Heart Rate (BPM) |
| SpO2 timeseries | Time (seconds) | SpO2 (%) — expect 95–100% for healthy subject |

---

*Document generated: March 2026 | AS7058 + SP-20 Dataset Analysis*
