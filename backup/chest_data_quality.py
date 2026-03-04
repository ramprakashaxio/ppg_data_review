"""
Chest PPG Data Quality Analysis
=================================
Analyses the filtered chest CSV file and answers:
  1. Which time segments are GOOD / FAIR / POOR quality?
  2. How long does the signal take to stabilise after power-on?
  3. Where are the best windows for algorithm development?

File: Chest_position_nikhil_V1_02.032026_2026-03-02_13-18-43_filtered.csv
      (pre-filtered: DC baseline already removed, only AC pulsatile component)

Output: output/chest_analysis/ folder
  01_full_signal_annotated.png   -- full 337-second signal, good/bad zones coloured
  02_quality_over_time.png       -- sliding AC amplitude, SNR, HR per 5-second window
  03_stabilisation_zoom.png      -- zoomed 0-30s showing AGC settling
  04_good_vs_bad_waveform.png    -- waveform samples from GOOD vs POOR windows

Run: py -3 chest_data_quality.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal as sp

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
FPATH  = os.path.join(BASE,
    'AS7058/03_Chest_AS7058/V1_wrist algo/'
    'Chest_position_nikhil_V1_02.032026_2026-03-02_13-18-43_filtered.csv')
OUT    = os.path.join(BASE, 'output', 'chest_analysis')
os.makedirs(OUT, exist_ok=True)

# ── Quality thresholds (AC amplitude-based, since DC is removed in this file) ─
# For filtered files AC amplitude is the only quality indicator we have
AC_GOOD  = 200   # ADC counts peak-to-peak  (robust heartbeat visible)
AC_FAIR  = 80    # ADC counts peak-to-peak  (weak but detectable)
# < 80 = POOR (noise only)

SNR_GOOD = 6.0   # dB
SNR_FAIR = 2.0   # dB

# ── Load ───────────────────────────────────────────────────────────────────────
print('Loading chest filtered CSV ...')
raw = pd.read_csv(FPATH, low_memory=False)

ts   = pd.to_numeric(raw['TIMESTAMP [s]'], errors='coerce').values
ir   = pd.to_numeric(raw['PPG1_SUB1'],     errors='coerce').values   # IR AC
red  = pd.to_numeric(raw['PPG1_SUB2'],     errors='coerce').values   # RED AC

# Keep only valid rows
valid = ~(np.isnan(ts) | np.isnan(ir))
ts, ir, red = ts[valid], ir[valid], red[valid]

# Sampling rate
diffs = np.diff(ts)
fs    = int(round(1.0 / np.median(diffs[diffs > 0])))
dur   = ts[-1]
print(f'  Duration: {dur:.1f}s   Fs: {fs} Hz   Rows: {len(ts)}')

# ── Sliding window quality metrics ────────────────────────────────────────────
WIN_S  = 5.0    # window size (seconds)
STEP_S = 1.0    # step size  (seconds)

win_times, ac_amps, snrs, hrs, n_peaks_list = [], [], [], [], []

t_start = 0.0
while t_start + WIN_S <= dur:
    t_end = t_start + WIN_S
    m     = (ts >= t_start) & (ts < t_end)
    chunk = ir[m]

    if len(chunk) < int(fs * 2):
        t_start += STEP_S
        continue

    # Remove any extreme outliers (clip to ±5*std)
    std_c  = np.std(chunk)
    chunk  = np.clip(chunk, -5 * std_c, 5 * std_c) if std_c > 0 else chunk

    # AC amplitude (90th-10th percentile range — robust)
    ac  = np.percentile(chunk, 90) - np.percentile(chunk, 10)

    # SNR via Welch PSD
    try:
        f, p   = sp.welch(chunk, fs=fs, nperseg=min(len(chunk), int(fs * 4)))
        s_pow  = np.trapz(p[(f >= 0.7) & (f <= 3.5)], f[(f >= 0.7) & (f <= 3.5)])
        n_pow  = np.trapz(p[(f >= 4.0) & (f <= 8.0)], f[(f >= 4.0) & (f <= 8.0)])
        snr    = 10 * np.log10(s_pow / n_pow) if n_pow > 0 and s_pow > 0 else 0
    except Exception:
        snr = 0

    # Peak detection for HR
    dist   = int(fs * 0.35)
    thr    = np.median(chunk) + 0.35 * np.std(chunk)
    peaks, _ = sp.find_peaks(chunk, distance=dist, height=thr)
    hr = 0.0
    if len(peaks) > 1:
        rr = np.diff(ts[m][peaks])
        hr = 60.0 / np.median(rr) if np.median(rr) > 0 else 0

    win_times.append(t_start + WIN_S / 2)
    ac_amps.append(ac)
    snrs.append(snr)
    hrs.append(hr)
    n_peaks_list.append(len(peaks))
    t_start += STEP_S

win_times   = np.array(win_times)
ac_amps     = np.array(ac_amps)
snrs        = np.array(snrs)
hrs         = np.array(hrs)

# Quality label per window
def label(ac, snr):
    if ac >= AC_GOOD and snr >= SNR_GOOD:  return 'GOOD'
    if ac >= AC_FAIR or  snr >= SNR_FAIR:  return 'FAIR'
    return 'POOR'

quality = np.array([label(a, s) for a, s in zip(ac_amps, snrs)])

# Stabilisation: first window where quality becomes GOOD or FAIR and stays so
# Use a 10-window rolling majority vote
stable_t = None
for i in range(len(quality) - 9):
    window_q = quality[i:i + 10]
    if np.sum(window_q != 'POOR') >= 7:   # 7 of 10 windows non-POOR
        stable_t = win_times[i]
        break

# Summary stats
good_pct = np.mean(quality == 'GOOD') * 100
fair_pct = np.mean(quality == 'FAIR') * 100
poor_pct = np.mean(quality == 'POOR') * 100

print(f'\n  Quality breakdown:')
print(f'    GOOD : {good_pct:.1f}%  (AC >= {AC_GOOD}, SNR >= {SNR_GOOD} dB)')
print(f'    FAIR : {fair_pct:.1f}%  (AC >= {AC_FAIR} or SNR >= {SNR_FAIR} dB)')
print(f'    POOR : {poor_pct:.1f}%')
if stable_t:
    print(f'    Signal stabilises at: t = {stable_t:.1f}s after recording start')
else:
    print(f'    Signal never fully stabilised (mostly POOR throughout)')

# ── Helpers ────────────────────────────────────────────────────────────────────
QCOL = {'GOOD': '#2E7D32', 'FAIR': '#F57F17', 'POOR': '#B71C1C'}

def shade_quality(ax, win_times, quality, y0, y1):
    """Shade background of axis with quality colour."""
    prev_q = quality[0]
    prev_t = win_times[0] - 0.5
    for i in range(1, len(win_times)):
        if quality[i] != prev_q or i == len(win_times) - 1:
            end_t = win_times[i - 1] + 0.5
            ax.axvspan(prev_t, end_t, ymin=0, ymax=1,
                       color=QCOL[prev_q], alpha=0.12, zorder=0)
            prev_q = quality[i]
            prev_t = win_times[i] - 0.5

# ── Plot 1: Full signal annotated ─────────────────────────────────────────────
print('\nGenerating plots ...')

# Downsample for display (plot every 4th point = 50 Hz equiv)
step  = max(1, fs // 50)
ts_d  = ts[::step]
ir_d  = ir[::step]

fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
fig.suptitle(
    'Chest PPG — Full Recording Quality Analysis\n'
    'Chest_position_nikhil_V1  |  Pre-filtered (AC only, DC removed)  |  337 seconds',
    fontsize=13, fontweight='bold'
)

# --- Panel 1: IR waveform
ax = axes[0]
ax.plot(ts_d, ir_d, color='#1565C0', lw=0.4, alpha=0.8)
shade_quality(ax, win_times, quality, ir_d.min(), ir_d.max())
ax.set_ylabel('IR AC\n(ADC counts)', fontsize=9)
ax.set_title('IR Channel (PPG1_SUB1) — AC waveform', fontsize=10, fontweight='bold', loc='left')
ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
ax.grid(True, alpha=0.2)
ax.set_ylim(np.percentile(ir_d, 0.5), np.percentile(ir_d, 99.5))

# --- Panel 2: AC amplitude over time
ax = axes[1]
ax.plot(win_times, ac_amps, color='#4A148C', lw=1.5)
shade_quality(ax, win_times, quality, 0, ac_amps.max())
ax.axhline(AC_GOOD, color='green', lw=1.2, ls='--', label=f'GOOD threshold ({AC_GOOD})')
ax.axhline(AC_FAIR, color='orange', lw=1.2, ls='--', label=f'FAIR threshold ({AC_FAIR})')
ax.set_ylabel('AC Amplitude\n(ADC counts)', fontsize=9)
ax.set_title('Pulsatile Amplitude (per 5-second window)', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)
ax.set_ylim(0, np.percentile(ac_amps, 99) * 1.3)

# --- Panel 3: SNR
ax = axes[2]
ax.plot(win_times, snrs, color='#E65100', lw=1.5)
shade_quality(ax, win_times, quality, snrs.min(), snrs.max())
ax.axhline(SNR_GOOD, color='green', lw=1.2, ls='--', label=f'GOOD threshold ({SNR_GOOD} dB)')
ax.axhline(SNR_FAIR, color='orange', lw=1.2, ls='--', label=f'FAIR threshold ({SNR_FAIR} dB)')
ax.set_ylabel('SNR (dB)', fontsize=9)
ax.set_title('Signal-to-Noise Ratio (cardiac band 0.7-3.5 Hz vs noise 4-8 Hz)', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)

# --- Panel 4: HR
ax = axes[3]
valid_hr = hrs > 30
ax.plot(win_times[valid_hr], hrs[valid_hr], 'o', color='#006064',
        ms=3, alpha=0.7, label='Detected HR')
ax.axhline(60, color='gray', lw=0.8, ls=':', alpha=0.6)
ax.axhline(100, color='gray', lw=0.8, ls=':', alpha=0.6)
shade_quality(ax, win_times, quality, 0, 200)
ax.set_ylabel('Heart Rate\n(BPM)', fontsize=9)
ax.set_title('Estimated Heart Rate from Peak Detection', fontsize=10, fontweight='bold', loc='left')
ax.set_ylim(30, 180)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xlabel('Time (seconds)', fontsize=10)

# Stabilisation line
if stable_t:
    for ax in axes:
        ax.axvline(stable_t, color='black', lw=1.5, ls='-', alpha=0.7, zorder=10)
    axes[0].text(stable_t + 1, axes[0].get_ylim()[1] * 0.85,
                 f'Stable at\nt={stable_t:.0f}s',
                 fontsize=8, color='black', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Quality legend
patches = [mpatches.Patch(color=QCOL[q], alpha=0.4, label=q) for q in ['GOOD', 'FAIR', 'POOR']]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.01), title='Background shading = signal quality')

plt.tight_layout(rect=[0, 0.04, 1, 1])
out1 = os.path.join(OUT, '01_full_signal_annotated.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> {out1}')

# ── Plot 2: Quality breakdown bar + timeline ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Signal Quality Summary — Chest V1 Filtered', fontsize=12, fontweight='bold')

# Pie / bar of quality split
ax = axes[0]
pcts  = [good_pct, fair_pct, poor_pct]
clrs  = ['#2E7D32', '#F57F17', '#B71C1C']
lbls  = [f'GOOD\n{good_pct:.1f}%', f'FAIR\n{fair_pct:.1f}%', f'POOR\n{poor_pct:.1f}%']
wedges, _ = ax.pie(pcts, colors=clrs, startangle=90,
                   wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2))
ax.legend(wedges, lbls, loc='center', fontsize=11)
ax.set_title(f'Quality Breakdown\n(Total {dur:.0f}s recording)', fontsize=11)

# AC amplitude histogram coloured by quality
ax = axes[1]
for q, c in QCOL.items():
    mask = quality == q
    if mask.any():
        ax.hist(ac_amps[mask], bins=30, color=c, alpha=0.65, label=q, edgecolor='white')
ax.axvline(AC_GOOD, color='green',  lw=1.5, ls='--', label=f'GOOD threshold ({AC_GOOD})')
ax.axvline(AC_FAIR, color='orange', lw=1.5, ls='--', label=f'FAIR threshold ({AC_FAIR})')
ax.set_xlabel('AC Amplitude (ADC counts, 5-second window)', fontsize=10)
ax.set_ylabel('Number of windows', fontsize=10)
ax.set_title('AC Amplitude Distribution', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out2 = os.path.join(OUT, '02_quality_breakdown.png')
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> {out2}')

# ── Plot 3: Stabilisation zoom (0-30 seconds) ─────────────────────────────────
ZOOM = 30.0
zm   = ts <= ZOOM
ts_z = ts[zm];   ir_z = ir[zm]

# Find large transient: where |ir| > 5x median of stable portion
stable_mask  = ts >= 20
stable_std   = np.std(ir[stable_mask]) if stable_mask.any() else 1
transient_end = ts_z[np.argmax(np.abs(ir_z) < 3 * stable_std)] if stable_std > 0 else 5.0

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle('AGC Settling / Stabilisation — First 30 Seconds\n'
             'Shows how long the chest sensor takes to produce clean PPG signal',
             fontsize=12, fontweight='bold')

# Panel 1: raw waveform 0-30s
ax = axes[0]
ax.plot(ts_z, ir_z, color='#1565C0', lw=0.6)
ax.set_ylabel('IR AC (ADC counts)', fontsize=9)
ax.set_title('IR Signal — full amplitude (AGC transient visible at start)', fontsize=10, loc='left')
ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
ax.grid(True, alpha=0.2)

# Panel 2: clipped to ±200 to see when settled signal appears
ax = axes[1]
ir_clip = np.clip(ir_z, -500, 500)
ax.plot(ts_z, ir_clip, color='#1565C0', lw=0.8)
ax.axhline(AC_GOOD / 2, color='green',  lw=1, ls='--', alpha=0.7, label=f'Expected GOOD amplitude ~{AC_GOOD//2}')
ax.axhline(-AC_GOOD / 2, color='green', lw=1, ls='--', alpha=0.7)
ax.set_ylabel('IR AC clipped\n(±500)', fontsize=9)
ax.set_title('IR Signal — clipped ±500 ADC counts (reveals heartbeat when stable)', fontsize=10, loc='left')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 3: rolling std as proxy for "noisiness"
roll_std = pd.Series(ir_z).rolling(window=int(fs * 0.5), center=True).std().values
ax = axes[2]
ax.plot(ts_z, roll_std, color='#B71C1C', lw=1.2, label='Rolling std (0.5s window)')
ax.axhline(AC_GOOD, color='green',  lw=1.2, ls='--', label=f'Signal amplitude threshold ({AC_GOOD})')
ax.axhline(AC_FAIR, color='orange', lw=1.2, ls='--', label=f'Minimum threshold ({AC_FAIR})')
ax.set_ylabel('Rolling Std\n(ADC counts)', fontsize=9)
ax.set_title('Signal Energy — drops as AGC settles, rises when heartbeat appears', fontsize=10, loc='left')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylim(0, np.percentile(roll_std[~np.isnan(roll_std)], 95) * 1.3)

if stable_t and stable_t <= ZOOM:
    for ax in axes:
        ax.axvline(stable_t, color='black', lw=2, ls='-', alpha=0.8)
    axes[0].text(stable_t + 0.3, axes[0].get_ylim()[1] * 0.7,
                 f'Stable at t={stable_t:.0f}s',
                 fontsize=9, color='black', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.85))

# Shade settling zone
for ax in axes:
    ax.axvspan(0, stable_t if stable_t else ZOOM,
               color='red', alpha=0.06, label='AGC settling zone')

plt.tight_layout()
out3 = os.path.join(OUT, '03_stabilisation_zoom.png')
fig.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> {out3}')

# ── Plot 4: GOOD vs POOR waveform samples ─────────────────────────────────────
# Find a GOOD window and a POOR window (from stable part)
good_wins = win_times[(quality == 'GOOD') & (win_times > (stable_t or 10))]
poor_wins = win_times[(quality == 'POOR') & (win_times > (stable_t or 10))]
fair_wins = win_times[(quality == 'FAIR') & (win_times > (stable_t or 10))]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
fig.suptitle('Sample Waveforms — GOOD vs FAIR vs POOR quality windows\n'
             '5-second excerpt from each quality zone', fontsize=12, fontweight='bold')

def plot_sample(ax, center_t, label, color):
    m   = (ts >= center_t - 2.5) & (ts < center_t + 2.5)
    t_s = ts[m] - ts[m][0]
    sig = ir[m]
    if len(sig) < 10:
        ax.text(0.5, 0.5, 'No data in this window', transform=ax.transAxes,
                ha='center', va='center')
        return
    ax.plot(t_s, sig, color=color, lw=1.2)
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.fill_between(t_s, 0, np.where(sig > 0, sig, 0), color=color, alpha=0.15)
    # Detect peaks
    dist_p = int(fs * 0.35)
    thr_p  = np.median(sig) + 0.3 * np.std(sig)
    peaks, _ = sp.find_peaks(sig, distance=dist_p, height=thr_p)
    if len(peaks):
        ax.plot(t_s[peaks], sig[peaks], 'v', color='orange', ms=8, zorder=5,
                label=f'{len(peaks)} beats detected')
    ac_a = np.percentile(sig, 90) - np.percentile(sig, 10)
    ax.set_title(f'{label}  |  t={center_t:.0f}s  |  AC amplitude={ac_a:.0f} ADC counts',
                 fontsize=10, fontweight='bold', color=color, loc='left')
    ax.set_ylabel('IR AC (ADC)', fontsize=8)
    ax.grid(True, alpha=0.2)
    if len(peaks):
        ax.legend(fontsize=8)
    ax.set_xlim(0, 5)

# Pick representative windows
sample_good = good_wins[len(good_wins) // 2] if len(good_wins) else None
sample_fair = fair_wins[len(fair_wins) // 2] if len(fair_wins) else None
sample_poor = poor_wins[len(poor_wins) // 2] if len(poor_wins) else None

if sample_good: plot_sample(axes[0], sample_good, 'GOOD quality window', '#2E7D32')
else:           axes[0].set_title('No GOOD window found')
if sample_fair: plot_sample(axes[1], sample_fair, 'FAIR quality window', '#F57F17')
else:           axes[1].set_title('No FAIR window found')
if sample_poor: plot_sample(axes[2], sample_poor, 'POOR quality window (noise only)', '#B71C1C')
else:           axes[2].set_title('No POOR window found')

axes[-1].set_xlabel('Time within window (seconds)', fontsize=9)

plt.tight_layout()
out4 = os.path.join(OUT, '04_good_vs_poor_waveform.png')
fig.savefig(out4, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> {out4}')

# ── Console summary ────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('CHEST PPG QUALITY SUMMARY')
print('=' * 65)
print(f'  File duration       : {dur:.0f} seconds')
print(f'  Sampling rate       : {fs} Hz')
print(f'  Total windows (5s)  : {len(quality)}')
print()
print(f'  AGC settling / stabilisation')
print(f'    Stable from       : t = {stable_t:.1f}s' if stable_t else '    Never stabilised fully')
print(f'    Usable recording  : {dur - stable_t:.0f}s  ({(dur-stable_t)/dur*100:.0f}% of total)' if stable_t else '')
print()
print(f'  Quality breakdown (after t={stable_t:.0f}s):' if stable_t else '  Quality breakdown (full):')
after = (win_times >= (stable_t or 0))
if after.any():
    g = np.mean(quality[after] == 'GOOD') * 100
    f = np.mean(quality[after] == 'FAIR') * 100
    p = np.mean(quality[after] == 'POOR') * 100
    print(f'    GOOD : {g:.1f}%')
    print(f'    FAIR : {f:.1f}%')
    print(f'    POOR : {p:.1f}%')
print()
print(f'  AC amplitude stats (after stable):')
ac_stable = ac_amps[after] if after.any() else ac_amps
print(f'    Median : {np.median(ac_stable):.0f} ADC counts')
print(f'    Max    : {np.max(ac_stable):.0f} ADC counts')
print(f'    Min    : {np.min(ac_stable):.0f} ADC counts')
print()
print(f'  SNR stats (after stable):')
snr_stable = snrs[after] if after.any() else snrs
print(f'    Median : {np.median(snr_stable):.1f} dB')
print(f'    Max    : {np.max(snr_stable):.1f} dB')
print()
print(f'  VERDICT: Chest V1 filtered file has {good_pct:.0f}% GOOD + {fair_pct:.0f}% FAIR windows.')
if good_pct + fair_pct >= 60:
    print('  -> Usable for HR algorithm development (>60% acceptable quality).')
else:
    print('  -> Limited usability. Chest signal is weak — needs better sensor placement.')
print('=' * 65)
print(f'\nAll plots saved to: {OUT}')
