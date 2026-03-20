# Position B Chest PPG Quality Report - Iteration 2

**Sensor:** AS7058 | **Sampling:** 200 Hz | **Date:** 04 March 2026

## Verdict: **MARGINAL**

Position B passes some thresholds but has a critical weakness.
**Best channel:** SpO2 SUB3

---
## Comparison with Iteration 1 Chest

| Metric | Iteration 1 | Position B (SpO2 SUB3) | Change |
|--------|-------------|------------------------|--------|
| PI % | 0.220% | 0.109% | -0.111% (DEGRADED, -50.5%) |
| SNR (dB) | 19.1 | 25.7 | +6.6 |
| AC Amp (cts) | 6 | 587 | +581 |

---
## Computed Metrics (median of last 120s)

| Metric | SpO2 RED | SpO2 SUB3 | HRM CH1 | SpO2F_PPG1_SUB1 | SpO2F_PPG1_SUB2 | SpO2F_PPG1_SUB3 | HRMF_PPG1_SUB1 | HRMF_PPG1_SUB2 | Iter1 Chest |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------------|
| PI % | 0.0776 | 0.1089 | 0.0166 | N/A | N/A | N/A | N/A | N/A | 0.2200 |
| SNR (dB) | 25.4 | 25.7 | 36.4 | 20.1 | 26.1 | 26.4 | 37.6 | 17.9 | 19.1 |
| AC Amp (cts) | 345.1 | 587.5 | 58.1 | 5.1 | 315.5 | 562.4 | 56.1 | 4.8 | 6.0 |
| HR (bpm) | 71.5 | 93.5 | 111.8 | 86.9 | 75.1 | 98.2 | 111.6 | 86.1 | N/A |
| HR Valid % | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RR CV % | 36.8 | 33.0 | 2.7 | 33.0 | 32.4 | 30.7 | 2.5 | 30.6 | 3.6 |

---
## Quality Grades

| Metric | SpO2 RED | SpO2 SUB3 | HRM CH1 | SpO2F_PPG1_SUB1 | SpO2F_PPG1_SUB2 | SpO2F_PPG1_SUB3 | HRMF_PPG1_SUB1 | HRMF_PPG1_SUB2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| PI | POOR | POOR | POOR | N/A | N/A | N/A | N/A | N/A |
| SNR | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD |
| AC Amp | GOOD | GOOD | FAIR | POOR | GOOD | GOOD | FAIR | POOR |
| HR Valid | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD | GOOD |
| RR CV | POOR | POOR | GOOD | POOR | POOR | POOR | GOOD | POOR |

---
## Sensor-Reported Metrics (AS7058 on-chip algorithm)

### SpO2 Mode

- **PI:** median=0.0900%, range=0.0000-0.2700%
- **Signal Quality:** median=40, range=0-100
- **HR:** median=85.7 bpm, range=52.1-133.3 bpm
- **SpO2 %:** median=94.3%, range=85.0-100.0%

### HRM Mode

- **Signal Quality:** median=54, range=42-91
- **HR:** median=112.4 bpm, range=46.6-114.2 bpm
- **Respiration Rate:** median=21.9 bpm, confidence median=87

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
- PI = 0.1089% is **BELOW** the minimum 0.3% threshold
- The pulsatile (AC) component is extremely weak relative to DC
- SNR = 25.7 dB: cardiac power is **much stronger** than noise floor

---
## Conclusion

Position B shows marginal signal quality. Basic heart rate detection may be possible in the HRM channel, but SpO2 measurement is not feasible.

**Recommendation:** Compare with other positions. Consider this position only if no better alternative is found.

---
*Report generated: 20 March 2026, 02:41 PM*