# Position D Chest PPG Quality Report - Iteration 2

**Sensor:** AS7058 | **Sampling:** 200 Hz | **Date:** 04 March 2026

## Verdict: **USABLE**

Position D meets minimum signal quality requirements.
**Best channel:** SpO2 RED

---
## Comparison with Iteration 1 Chest

| Metric | Iteration 1 | Position D (SpO2 RED) | Change |
|--------|-------------|------------------------|--------|
| PI % | 0.220% | 0.418% | +0.198% (IMPROVED, +89.8%) |
| SNR (dB) | 19.1 | 27.7 | +8.6 |
| AC Amp (cts) | 6 | 1614 | +1608 |

---
## Computed Metrics (median of last 120s)

| Metric | SpO2 RED | SpO2 SUB3 | HRM CH1 | SpO2F_PPG1_SUB1 | SpO2F_PPG1_SUB2 | SpO2F_PPG1_SUB3 | HRMF_PPG1_SUB1 | HRMF_PPG1_SUB2 | Iter1 Chest |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------------|
| PI % | 0.4177 | 0.2697 | 0.0181 | N/A | N/A | N/A | N/A | N/A | 0.2200 |
| SNR (dB) | 27.7 | 27.4 | 28.7 | 18.4 | 28.2 | 27.6 | 29.6 | 17.9 | 19.1 |
| AC Amp (cts) | 1613.8 | 1941.8 | 64.7 | 4.4 | 1272.6 | 1713.3 | 58.8 | 4.8 | 6.0 |
| HR (bpm) | 62.7 | 64.5 | 105.6 | 87.9 | 64.5 | 67.2 | 109.9 | 91.1 | N/A |
| HR Valid % | 83.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RR CV % | 31.4 | 32.0 | 27.1 | 30.7 | 32.4 | 34.3 | 22.5 | 32.5 | 3.6 |

---
## Quality Grades

| Metric | SpO2 RED | SpO2 SUB3 | HRM CH1 | SpO2F_PPG1_SUB1 | SpO2F_PPG1_SUB2 | SpO2F_PPG1_SUB3 | HRMF_PPG1_SUB1 | HRMF_PPG1_SUB2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| PI | FAIR | POOR | POOR | N/A | N/A | N/A | N/A | N/A |
| SNR | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD |
| AC Amp | GOOD | GOOD | FAIR | POOR | GOOD | GOOD | FAIR | POOR |
| HR Valid | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD |
| RR CV | POOR | POOR | POOR | POOR | POOR | POOR | FAIR | POOR |

---
## Sensor-Reported Metrics (AS7058 on-chip algorithm)

### SpO2 Mode

- **PI:** median=0.3200%, range=0.0000-2.7900%
- **Signal Quality:** median=60, range=0-100
- **HR:** median=62.3 bpm, range=37.5-96.0 bpm
- **SpO2 %:** median=85.5%, range=82.4-100.0%

### HRM Mode

- **Signal Quality:** median=36, range=27-91
- **HR:** median=113.8 bpm, range=48.0-117.5 bpm
- **Respiration Rate:** median=19.8 bpm, confidence median=91

---
## AGC (Automatic Gain Control) Analysis

| Mode | Channel | LED Current (LSB) | Grade |
|------|---------|-------------------|-------|
| SpO2 | AGC1_led | 160 | POOR |
| SpO2 | AGC2_led | 120 | POOR |
| HRM | AGC1_led | 20 | GOOD |

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
- PI = 0.4177% is **within acceptable range**
- SNR = 27.7 dB: cardiac power is **much stronger** than noise floor

---
## Conclusion

Position D meets minimum signal quality thresholds. Heart rate detection is feasible. SpO2 may be possible with algorithm optimization.

---
*Report generated: 20 March 2026, 02:41 PM*