# Position A Chest PPG Quality Report - Iteration 2

**Sensor:** AS7058 | **Sampling:** 200 Hz | **Date:** 04 March 2026

## Verdict: **USABLE**

Position A meets minimum signal quality requirements.
**Best channel:** HRM CH1

---
## Comparison with Iteration 1 Chest

| Metric | Iteration 1 | Position A (HRM CH1) | Change |
|--------|-------------|------------------------|--------|
| PI % | 0.220% | 0.660% | +0.440% (IMPROVED, +200.0%) |
| SNR (dB) | 19.1 | 39.0 | +19.9 |
| AC Amp (cts) | 6 | 3721 | +3715 |

---
## Computed Metrics (median of last 120s)

| Metric | SpO2 RED | SpO2 SUB3 | HRM CH1 | SpO2F_PPG1_SUB1 | SpO2F_PPG1_SUB2 | SpO2F_PPG1_SUB3 | HRMF_PPG1_SUB1 | HRMF_PPG1_SUB2 | Iter1 Chest |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------------|
| PI % | 0.0593 | 0.0897 | 0.6599 | N/A | N/A | N/A | N/A | N/A | 0.2200 |
| SNR (dB) | 31.8 | 34.0 | 39.0 | 18.8 | 33.3 | 35.1 | 40.0 | 19.0 | 19.1 |
| AC Amp (cts) | 391.1 | 628.2 | 3720.9 | 4.6 | 383.4 | 621.9 | 3312.0 | 5.1 | 6.0 |
| HR (bpm) | 97.6 | 100.9 | 94.3 | 93.5 | 98.4 | 104.2 | 99.1 | 85.0 | N/A |
| HR Valid % | 100.0 | 100.0 | 86.9 | 100.0 | 100.0 | 100.0 | 100.0 | 91.8 | 100.0 |
| RR CV % | 26.1 | 4.3 | 25.0 | 34.6 | 23.1 | 3.9 | 7.0 | 32.1 | 3.6 |

---
## Quality Grades

| Metric | SpO2 RED | SpO2 SUB3 | HRM CH1 | SpO2F_PPG1_SUB1 | SpO2F_PPG1_SUB2 | SpO2F_PPG1_SUB3 | HRMF_PPG1_SUB1 | HRMF_PPG1_SUB2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| PI | POOR | POOR | FAIR | N/A | N/A | N/A | N/A | N/A |
| SNR | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD |
| AC Amp | GOOD | GOOD | GOOD | POOR | GOOD | GOOD | GOOD | POOR |
| HR Valid | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD |
| RR CV | POOR | GOOD | FAIR | POOR | FAIR | GOOD | GOOD | POOR |

---
## Sensor-Reported Metrics (AS7058 on-chip algorithm)

### SpO2 Mode

- **PI:** median=0.0700%, range=0.0000-0.0900%
- **Signal Quality:** median=80, range=0-100
- **HR:** median=93.3 bpm, range=61.5-107.1 bpm
- **SpO2 %:** median=91.5%, range=88.1-100.0%

### HRM Mode

- **Signal Quality:** median=30, range=24-91
- **HR:** median=101.5 bpm, range=63.4-109.5 bpm
- **Respiration Rate:** median=19.5 bpm, confidence median=82

---
## AGC (Automatic Gain Control) Analysis

| Mode | Channel | LED Current (LSB) | Grade |
|------|---------|-------------------|-------|
| SpO2 | AGC1_led | 160 | POOR |
| SpO2 | AGC2_led | 120 | POOR |
| HRM | AGC1_led | 48 | FAIR |

---
## Signal vs Noise Breakdown

### Signal (what we want)
- **Cardiac pulsation:** blood volume changes with each heartbeat
- **Frequency range:** 0.7 - 3.5 Hz (42 - 210 bpm)
- Appears as periodic waveform in bandpass-filtered PPG

### Noise (what degrades measurement)
- **High-frequency noise (>4 Hz):** electronic/optical noise
- **Baseline drift (<0.5 Hz):** sensor movement, temperature
- **Motion artifacts:** body/sensor movement during measurement
- **Low perfusion:** insufficient blood flow at measurement site

### At this chest position
- PI = 0.6599% is **within acceptable range**
- SNR = 39.0 dB: cardiac power is **much stronger** than noise floor

---
## Conclusion

Position A meets minimum signal quality thresholds. Heart rate detection is feasible. SpO2 may be possible with algorithm optimization.

---
*Report generated: 20 March 2026, 02:20 PM*