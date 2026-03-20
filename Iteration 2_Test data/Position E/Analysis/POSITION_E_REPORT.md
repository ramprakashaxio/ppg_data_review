# Position E Chest PPG Quality Report - Iteration 2

**Sensor:** AS7058 | **Sampling:** 200 Hz | **Date:** 04 March 2026

## Verdict: **MARGINAL**

Position E passes some thresholds but has a critical weakness.
**Best channel:** SpO2 SUB3

---
## Comparison with Iteration 1 Chest

| Metric | Iteration 1 | Position E (SpO2 SUB3) | Change |
|--------|-------------|------------------------|--------|
| PI % | 0.220% | 0.107% | -0.113% (DEGRADED, -51.3%) |
| SNR (dB) | 19.1 | 32.2 | +13.1 |
| AC Amp (cts) | 6 | 424 | +418 |

---
## Computed Metrics (median of last 120s)

| Metric | SpO2 RED | SpO2 SUB3 | HRM CH1 | SpO2F_PPG1_SUB1 | SpO2F_PPG1_SUB2 | SpO2F_PPG1_SUB3 | HRMF_PPG1_SUB1 | HRMF_PPG1_SUB2 | Iter1 Chest |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------------|
| PI % | 0.0482 | 0.1070 | 0.0255 | N/A | N/A | N/A | N/A | N/A | 0.2200 |
| SNR (dB) | 30.7 | 32.2 | 39.0 | 19.7 | 31.1 | 32.4 | 40.5 | 19.3 | 19.1 |
| AC Amp (cts) | 271.0 | 424.2 | 93.0 | 5.1 | 262.0 | 411.3 | 92.1 | 5.9 | 6.0 |
| HR (bpm) | 78.9 | 87.4 | 110.3 | 87.0 | 80.6 | 89.3 | 110.1 | 85.1 | N/A |
| HR Valid % | 95.1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RR CV % | 35.1 | 32.4 | 1.6 | 33.2 | 34.9 | 28.6 | 1.5 | 32.6 | 3.6 |

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

- **PI:** median=0.0600%, range=0.0000-2.3700%
- **Signal Quality:** median=20, range=0-100
- **HR:** median=54.5 bpm, range=30.3-133.3 bpm
- **SpO2 %:** median=95.2%, range=85.9-100.0%

### HRM Mode

- **Signal Quality:** median=72, range=55-91
- **HR:** median=110.3 bpm, range=46.2-117.1 bpm
- **Respiration Rate:** median=21.1 bpm, confidence median=92

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
- PI = 0.1070% is **BELOW** the minimum 0.3% threshold
- The pulsatile (AC) component is extremely weak relative to DC
- SNR = 32.2 dB: cardiac power is **much stronger** than noise floor

---
## Conclusion

Position E shows marginal signal quality. Basic heart rate detection may be possible in the HRM channel, but SpO2 measurement is not feasible.

**Recommendation:** Compare with other positions. Consider this position only if no better alternative is found.

---
*Report generated: 20 March 2026, 02:41 PM*