"""
Chest PPG — All 4 Files Quality Analysis
==========================================
Analyses all 4 chest recordings (V1 raw, V1 filtered, V2 raw, V2 filtered)
and computes every meaningful signal quality metric.

WHY EACH METRIC MATTERS
------------------------
  PI %          - Perfusion Index. AC pulse amplitude / DC baseline x100.
                  Clinical standard: >1% = good, >0.3% = usable, <0.3% = poor.
                  Needs raw file (DC available).

  AC Amplitude  - Peak-to-peak size of the heartbeat ripple (ADC counts).
                  Directly tells you if the heart pulse is detectable.

  SNR (dB)      - How much stronger is the cardiac signal than background noise.
                  >6 dB = usable for HR.  >10 dB = good for SpO2.

  HR (BPM)      - Heart rate extracted from peaks. Should be 50-120 BPM.
                  Unrealistic values = noisy signal.

  RR Irregularity- Coefficient of variation of beat-to-beat intervals.
                  Low (<10%) = clean rhythm detected.
                  High (>30%) = false peaks / motion.

  DC Stability  - Std of raw baseline over time (raw files only).
                  High DC drift = sensor lifting off skin / motion artifact.

  Motion Index  - Std of accelerometer magnitude (raw files only).
                  Correlates directly with motion artifact in PPG.

  Sensor PI %   - On-chip reported PI from SPO2: PI [%] column (raw files).
                  Cross-check against computed PI.

Output: output/chest_all/ folder  (7 plots)
Run:    py -3 chest_all_quality.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import signal as sp

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'output', 'chest_all')
os.makedirs(OUT, exist_ok=True)

# ── File definitions ──────────────────────────────────────────────────────────
FILES = {
    'V1 Raw': {
        'path':   'AS7058/03_Chest_AS7058/V1_wrist algo/Chest_position_nikhil_V1_02.032026_2026-03-02_13-24-19.csv',
        'type':   'raw',
        'color':  '#C62828',
        'marker': 'o',
    },
    'V1 Filtered': {
        'path':   'AS7058/03_Chest_AS7058/V1_wrist algo/Chest_position_nikhil_V1_02.032026_2026-03-02_13-18-43_filtered.csv',
        'type':   'filtered',
        'color':  '#EF9A9A',
        'marker': 's',
    },
    'V2 Raw': {
        'path':   'AS7058/03_Chest_AS7058/V2_wrist algo/Chest_position_nikhil_V2_02.032026_2026-03-02_13-36-02.csv',
        'type':   'raw',
        'color':  '#1565C0',
        'marker': '^',
    },
    'V2 Filtered': {
        'path':   'AS7058/03_Chest_AS7058/V2_wrist algo/Chest_position_nikhil_V2_02.032026_2026-03-02_13-30-43_filtered.csv',
        'type':   'filtered',
        'color':  '#90CAF9',
        'marker': 'D',
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def bandpass(sig, fs, lo=0.5, hi=4.0, order=4):
    nyq = fs / 2.0
    b, a = sp.butter(order, [lo/nyq, min(hi/nyq, 0.99)], btype='band')
    return sp.filtfilt(b, a, sig)

def snr_db(sig, fs):
    f, p = sp.welch(np.nan_to_num(sig), fs=fs, nperseg=min(len(sig), int(fs*8)))
    sp_ = np.trapz(p[(f>=0.7)&(f<=3.5)], f[(f>=0.7)&(f<=3.5)])
    np_ = np.trapz(p[(f>=4.0)&(f<=8.0)], f[(f>=4.0)&(f<=8.0)])
    return 10*np.log10(sp_/np_) if np_>0 and sp_>0 else 0.0

def detect_hr(sig, ts_seg, fs):
    dist   = int(fs * 0.35)
    thr    = np.nanmedian(sig) + 0.35*np.nanstd(sig)
    peaks, _ = sp.find_peaks(sig, distance=dist, height=thr)
    if len(peaks) < 2:
        return 0.0, 99.0      # hr, rr_cv
    rr = np.diff(ts_seg[peaks])
    rr = rr[(rr>0.3)&(rr<2.0)]  # physiological range
    if len(rr) == 0:
        return 0.0, 99.0
    hr   = 60.0 / np.median(rr)
    rrcv = (np.std(rr) / np.mean(rr) * 100) if np.mean(rr)>0 else 99.0
    return hr, rrcv

# ── Load all files ────────────────────────────────────────────────────────────
print('Loading all 4 chest files ...')
datasets = {}

for name, cfg in FILES.items():
    path = os.path.join(BASE, cfg['path'])
    raw  = pd.read_csv(path, low_memory=False)
    ftype = cfg['type']

    ts   = pd.to_numeric(raw['TIMESTAMP [s]'],  errors='coerce')
    s1   = pd.to_numeric(raw['PPG1_SUB1'],       errors='coerce')
    s2   = pd.to_numeric(raw['PPG1_SUB2'],       errors='coerce')

    if ftype == 'raw':
        # PPG rows = where SUB1 is a large DC value (>1000)
        ppg_mask = s1.notna() & (s1 > 1000)
        # ACC rows
        acc_x    = pd.to_numeric(raw.get('ACC_X', pd.Series(dtype=float)), errors='coerce')
        acc_y    = pd.to_numeric(raw.get('ACC_Y', pd.Series(dtype=float)), errors='coerce')
        acc_z    = pd.to_numeric(raw.get('ACC_Z', pd.Series(dtype=float)), errors='coerce')
        acc_mask = acc_x.notna()
        # Sensor SpO2 events
        sq = pd.to_numeric(raw.get('SPO2: SIGNAL_QUALITY', pd.Series(dtype=float)), errors='coerce')
        pi_sens  = pd.to_numeric(raw.get('SPO2: PI [%]',  pd.Series(dtype=float)), errors='coerce')
        hr_sens  = pd.to_numeric(raw.get('SPO2: HEART_RATE [bpm]', pd.Series(dtype=float)), errors='coerce')
        # AGC
        agc_cols = [c for c in raw.columns if 'AGC1' in c and 'CURRENT' in c]
        agc_vals = pd.to_numeric(raw[agc_cols[0]], errors='coerce') if agc_cols else pd.Series(dtype=float)
        agc      = agc_vals.dropna().iloc[-1] if not agc_vals.dropna().empty else np.nan
    else:
        ppg_mask = s1.notna()
        acc_x = acc_y = acc_z = pd.Series(dtype=float)
        acc_mask = pd.Series([False]*len(raw))
        sq = pi_sens = hr_sens = pd.Series(dtype=float)
        agc = np.nan

    ppg = raw[ppg_mask].copy()
    t   = pd.to_numeric(ppg['TIMESTAMP [s]'], errors='coerce').values
    ir  = pd.to_numeric(ppg['PPG1_SUB1'],      errors='coerce').values
    red = pd.to_numeric(ppg['PPG1_SUB2'],      errors='coerce').values

    # Sort by time
    idx  = np.argsort(t)
    t, ir, red = t[idx], ir[idx], red[idx]
    t   = t - t[0]   # reset to 0

    # Mask zero RED (AGC settling) for raw files
    if ftype == 'raw':
        red = np.where(red == 0, np.nan, red.astype(float))

    dt = np.diff(t)
    fs = int(round(1.0 / np.median(dt[dt>0])))

    # For raw: DC stability and ACC motion
    if ftype == 'raw':
        dc_rolling = pd.Series(ir).rolling(window=int(fs*2), center=True).mean().values
        dc_drift   = pd.Series(ir).rolling(window=int(fs*2), center=True).std().values
        acc_data   = raw[acc_mask].copy()
        if len(acc_data) > 0:
            ax_ = pd.to_numeric(acc_data['ACC_X'], errors='coerce').values
            ay_ = pd.to_numeric(acc_data['ACC_Y'], errors='coerce').values
            az_ = pd.to_numeric(acc_data['ACC_Z'], errors='coerce').values
            acc_mag   = np.sqrt(ax_**2 + ay_**2 + az_**2)
            acc_t     = pd.to_numeric(acc_data['TIMESTAMP [s]'], errors='coerce').values
            acc_t     = acc_t - acc_t[0]
        else:
            acc_mag, acc_t = np.array([]), np.array([])
    else:
        dc_rolling = dc_drift = acc_mag = acc_t = np.array([])

    datasets[name] = dict(
        t=t, ir=ir, red=red, fs=fs, ftype=ftype, agc=agc,
        dc_drift=dc_drift, acc_mag=acc_mag, acc_t=acc_t,
        color=cfg['color'], marker=cfg['marker'],
        pi_sens=pi_sens, hr_sens=hr_sens,
        duration=t[-1],
    )
    print(f'  {name:<14}  {ftype:<8}  dur={t[-1]:.0f}s  fs={fs}Hz  rows={len(t)}  agc={agc:.0f}' if not np.isnan(agc) else
          f'  {name:<14}  {ftype:<8}  dur={t[-1]:.0f}s  fs={fs}Hz  rows={len(t)}')


# ── Sliding window metrics ────────────────────────────────────────────────────
WIN_S  = 10.0
STEP_S = 2.0

print('\nComputing sliding window metrics ...')
for name, ds in datasets.items():
    t, ir, fs, ftype = ds['t'], ds['ir'], ds['fs'], ds['ftype']
    win_t, pis, snrs, hrs, rrcvs, ac_amps, dc_stabs = [], [], [], [], [], [], []

    start = 5.0   # skip first 5s AGC transient
    while start + WIN_S <= ds['duration']:
        end  = start + WIN_S
        m    = (t >= start) & (t < end)
        seg  = ir[m]
        ts_s = t[m]

        if len(seg) < fs * 3:
            start += STEP_S; continue

        # Clip extreme outliers
        p1, p99 = np.percentile(seg, 1), np.percentile(seg, 99)
        seg_c   = np.clip(seg, p1, p99)

        if ftype == 'raw':
            # Bandpass to get AC from raw
            ac_sig = bandpass(seg_c - np.mean(seg_c), fs)
            dc_val = np.mean(seg_c)
            ac_ptp = np.percentile(ac_sig, 90) - np.percentile(ac_sig, 10)
            pi     = abs(ac_ptp) / abs(dc_val) * 100 if dc_val > 0 else 0
            dc_stab= np.std(seg_c) / np.mean(seg_c) * 100  # DC variation %
        else:
            # Already AC-only
            ac_sig  = seg_c
            ac_ptp  = np.percentile(ac_sig, 90) - np.percentile(ac_sig, 10)
            pi      = 0   # can't compute without DC
            dc_stab = 0

        snr      = snr_db(ac_sig - np.mean(ac_sig), fs)
        hr, rrcv = detect_hr(ac_sig, ts_s, fs)

        win_t.append(start + WIN_S/2)
        pis.append(pi)
        snrs.append(snr)
        hrs.append(hr)
        rrcvs.append(rrcv)
        ac_amps.append(ac_ptp)
        dc_stabs.append(dc_stab)
        start += STEP_S

    ds['win_t']    = np.array(win_t,    dtype=float)
    ds['pis']      = np.array(pis,      dtype=float)
    ds['snrs']     = np.array(snrs,     dtype=float)
    ds['hrs']      = np.array(hrs,      dtype=float)
    ds['rrcvs']    = np.array(rrcvs,    dtype=float)
    ds['ac_amps']  = np.array(ac_amps,  dtype=float)
    ds['dc_stabs'] = np.array(dc_stabs, dtype=float)

    valid_hr  = (ds['hrs'] > 40) & (ds['hrs'] < 160)
    pis_a = ds['pis']
    print(f'  {name:<14}  median_PI={np.median(pis_a[pis_a>0]):.2f}%  median_SNR={np.median(ds["snrs"]):.1f}dB  '
          f'median_HR={np.median(ds["hrs"][valid_hr]):.0f}bpm  valid_windows={valid_hr.sum()}/{len(ds["hrs"])}')


# ── PLOT 1: Waveform overview — all 4 files ───────────────────────────────────
print('\nGenerating plots ...')

fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=False)
fig.suptitle('Chest PPG — Waveform Overview: All 4 Files\n'
             'IR channel, first 60 seconds shown (bandpass 0.5-4 Hz for raw files)',
             fontsize=12, fontweight='bold')

for ax, (name, ds) in zip(axes, datasets.items()):
    t, ir, fs = ds['t'], ds['ir'], ds['fs']
    m = t <= 60
    t_s, ir_s = t[m], ir[m]

    if ds['ftype'] == 'raw':
        sig = bandpass(ir_s - np.mean(ir_s), fs)
        ax.set_title(f'{name}  (RAW → bandpass filtered for display)  '
                     f'duration={ds["duration"]:.0f}s  fs={fs}Hz  AGC={ds["agc"]:.0f}',
                     fontsize=9, fontweight='bold', color=ds['color'], loc='left')
    else:
        sig = ir_s
        ax.set_title(f'{name}  (Pre-filtered, AC-only)  '
                     f'duration={ds["duration"]:.0f}s  fs={fs}Hz',
                     fontsize=9, fontweight='bold', color=ds['color'], loc='left')

    ax.plot(t_s, sig, color=ds['color'], lw=0.5, alpha=0.85)
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax.set_ylabel('IR AC\n(ADC)', fontsize=8)
    ax.grid(True, alpha=0.18)
    lim = np.percentile(np.abs(sig), 98) * 1.4
    ax.set_ylim(-lim, lim)

axes[-1].set_xlabel('Time (seconds)', fontsize=10)
plt.tight_layout()
p = os.path.join(OUT, '01_waveform_overview.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'  Saved -> {p}')


# ── PLOT 2: SNR over time — all 4 files ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
for name, ds in datasets.items():
    ax.plot(ds['win_t'], ds['snrs'], color=ds['color'], lw=1.5,
            marker=ds['marker'], ms=3, alpha=0.8, label=name)
ax.axhline(10, color='green',  lw=1.5, ls='--', alpha=0.8, label='SNR 10 dB (SpO2 reliable)')
ax.axhline(6,  color='orange', lw=1.5, ls='--', alpha=0.8, label='SNR 6 dB (HR usable)')
ax.axhline(2,  color='red',    lw=1.5, ls='--', alpha=0.8, label='SNR 2 dB (noise floor)')
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylabel('SNR (dB)', fontsize=10)
ax.set_title('Signal-to-Noise Ratio Over Time — All 4 Chest Files\n'
             'Cardiac band (0.7-3.5 Hz) power / noise band (4-8 Hz) power',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.2)
plt.tight_layout()
p = os.path.join(OUT, '02_snr_over_time.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'  Saved -> {p}')


# ── PLOT 3: PI% over time (raw files only) ────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle('Perfusion Index (PI%) Over Time — Raw Files\n'
             'PI% = AC heartbeat amplitude / DC baseline x100',
             fontsize=11, fontweight='bold')

raw_ds = {n: d for n, d in datasets.items() if d['ftype'] == 'raw'}
for ax, (name, ds) in zip(axes, raw_ds.items()):
    ax.fill_between(ds['win_t'], 0, ds['pis'], color=ds['color'], alpha=0.3)
    ax.plot(ds['win_t'], ds['pis'], color=ds['color'], lw=1.5, label=f'{name} (computed)')
    # Sensor-reported PI
    pi_ev = ds['pi_sens'].dropna()
    ts_ev = pd.to_numeric(pd.Series(range(len(ds['pi_sens']))), errors='coerce')
    # Re-map sensor events to time (approximate)
    raw_path = os.path.join(BASE, FILES[name]['path'])
    raw_df   = pd.read_csv(raw_path, low_memory=False)
    sq_mask  = pd.to_numeric(raw_df.get('SPO2: SIGNAL_QUALITY', pd.Series(dtype=float)), errors='coerce').notna()
    pi_rows  = raw_df[sq_mask].copy()
    pi_t     = pd.to_numeric(pi_rows['TIMESTAMP [s]'], errors='coerce').values
    pi_v     = pd.to_numeric(pi_rows.get('SPO2: PI [%]', pd.Series(dtype=float)), errors='coerce').values
    if len(pi_t) > 0:
        pi_t = pi_t - pi_t[0]
        valid = ~np.isnan(pi_v)
        ax.plot(pi_t[valid], pi_v[valid], 'k.', ms=2, alpha=0.4, label='Sensor-reported PI')

    ax.axhline(1.0, color='green',  lw=1.2, ls='--', alpha=0.7, label='PI 1.0% (GOOD)')
    ax.axhline(0.3, color='orange', lw=1.2, ls='--', alpha=0.7, label='PI 0.3% (FAIR)')
    ax.set_ylabel('PI %', fontsize=9)
    ax.set_title(name, fontsize=10, fontweight='bold', color=ds['color'], loc='left')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)
    ax.set_ylim(0, max(np.percentile(ds['pis'], 98) * 1.3, 1.5))

axes[-1].set_xlabel('Time (seconds)', fontsize=10)
plt.tight_layout()
p = os.path.join(OUT, '03_pi_over_time.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'  Saved -> {p}')


# ── PLOT 4: AC Amplitude over time — all 4 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
for name, ds in datasets.items():
    ax.plot(ds['win_t'], ds['ac_amps'], color=ds['color'], lw=1.4,
            marker=ds['marker'], ms=3, alpha=0.8, label=name)
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylabel('AC Amplitude (ADC counts, 10-second window)', fontsize=10)
ax.set_title('Pulsatile (AC) Amplitude Over Time — All 4 Chest Files\n'
             'Peak-to-peak size of the heartbeat ripple — larger = stronger pulse signal',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
plt.tight_layout()
p = os.path.join(OUT, '04_ac_amplitude_over_time.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'  Saved -> {p}')


# ── PLOT 5: Heart Rate over time — all 4 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
for name, ds in datasets.items():
    valid = (ds['hrs'] > 40) & (ds['hrs'] < 160)
    ax.scatter(ds['win_t'][valid], ds['hrs'][valid], color=ds['color'],
               marker=ds['marker'], s=18, alpha=0.7, label=name)
ax.axhline(60,  color='gray', lw=0.8, ls=':')
ax.axhline(100, color='gray', lw=0.8, ls=':')
ax.set_ylim(30, 160)
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylabel('Detected HR (BPM)', fontsize=10)
ax.set_title('Heart Rate Detected From Peaks — All 4 Chest Files\n'
             'Consistent HR in 50-120 BPM = signal good enough for HR algorithm',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
plt.tight_layout()
p = os.path.join(OUT, '05_hr_over_time.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'  Saved -> {p}')


# ── PLOT 6: Comprehensive metrics summary (scorecard) ─────────────────────────
metric_names = ['Median\nPI %', 'Median\nSNR (dB)', 'Median\nAC Amp',
                'HR Valid\nWindows %', 'Median\nRR CV %', 'AGC\nCurrent']

def get_scores(ds):
    valid_hr = (ds['hrs'] > 40) & (ds['hrs'] < 160)
    return [
        np.median(ds['pis'][ds['pis'] > 0]) if (ds['pis'] > 0).any() else 0,
        np.median(ds['snrs']),
        np.median(ds['ac_amps']),
        valid_hr.mean() * 100,
        np.median(ds['rrcvs'][valid_hr]) if valid_hr.any() else 99,
        ds['agc'] if not np.isnan(ds['agc']) else 0,
    ]

scores     = {name: get_scores(ds) for name, ds in datasets.items()}
thresholds = [1.0, 6.0, None, 50.0, 20.0, None]   # good threshold per metric
higher_is_better = [True, True, True, True, False, False]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Signal Quality Scorecard — All 4 Chest Files\n'
             'Each metric explained: what is good/bad and why it matters',
             fontsize=12, fontweight='bold')
axes = axes.flatten()

explanations = [
    'PI% = AC/DC x100\n>1% = GOOD,  >0.3% = FAIR\nNeeds good skin contact',
    'SNR = cardiac power / noise power\n>10dB = SpO2 usable\n>6dB = HR usable',
    'Peak-to-peak heartbeat ripple\nHigher = stronger heart signal\nNo fixed threshold',
    'What % of windows gave\na physiological HR (40-160 BPM)\nHigher = more reliable',
    'Beat-to-beat interval variation\n<10% = clean signal\n>30% = noisy / false peaks',
    'LED drive current (lower=better)\nLow = sensor well-coupled to skin\nHigh = sensor floating',
]
ylabels = ['PI %', 'SNR (dB)', 'AC Amplitude (ADC)', '% windows', 'RR CV %', 'AGC LSB']

for i, ax in enumerate(axes):
    names = list(scores.keys())
    vals  = [scores[n][i] for n in names]
    cols  = [datasets[n]['color'] for n in names]

    bars = ax.bar(range(len(names)), vals, color=cols, edgecolor='white',
                  linewidth=1.0, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(ylabels[i], fontsize=8)
    ax.set_title(metric_names[i].replace('\n', ' '), fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)

    # Threshold line
    if thresholds[i] is not None:
        color_thr = 'green' if higher_is_better[i] else 'red'
        ax.axhline(thresholds[i], color=color_thr, lw=1.5, ls='--', alpha=0.8,
                   label=f'Threshold = {thresholds[i]}')
        ax.legend(fontsize=7)

    # Value labels on bars
    max_v = max(vals) if max(vals) > 0 else 1
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max_v * 0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Explanation box at bottom
    ax.text(0.01, -0.38, explanations[i], transform=ax.transAxes,
            fontsize=6.5, va='top', style='italic', color='#555555',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.7))

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4)
p = os.path.join(OUT, '06_quality_scorecard.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'  Saved -> {p}')


# ── PLOT 7: Motion vs signal quality (raw files, has ACC) ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Motion Artifact vs Signal Quality — Raw Files\n'
             'Top: accelerometer magnitude | Bottom: SNR at same time',
             fontsize=11, fontweight='bold')

for col, (name, ds) in enumerate(raw_ds.items()):
    ax_top = axes[0][col]
    ax_bot = axes[1][col]

    if len(ds['acc_t']) > 0:
        ax_top.plot(ds['acc_t'], ds['acc_mag'], color=ds['color'], lw=0.6, alpha=0.7)
        ax_top.set_ylabel('Acc magnitude\n(ADC counts)', fontsize=8)
    else:
        ax_top.text(0.5, 0.5, 'No ACC data', transform=ax_top.transAxes, ha='center')
    ax_top.set_title(f'{name} — Accelerometer', fontsize=9, fontweight='bold', color=ds['color'])
    ax_top.grid(True, alpha=0.2)

    ax_bot.plot(ds['win_t'], ds['snrs'], color=ds['color'], lw=1.4, label='SNR')
    ax_bot.axhline(6, color='orange', lw=1.2, ls='--', alpha=0.8, label='6 dB threshold')
    ax_bot.set_ylabel('SNR (dB)', fontsize=8)
    ax_bot.set_xlabel('Time (seconds)', fontsize=8)
    ax_bot.set_title(f'{name} — SNR', fontsize=9, fontweight='bold', color=ds['color'])
    ax_bot.legend(fontsize=7); ax_bot.grid(True, alpha=0.2)

plt.tight_layout()
p = os.path.join(OUT, '07_motion_vs_snr.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); print(f'  Saved -> {p}')


# ── Console summary ───────────────────────────────────────────────────────────
sep = '=' * 78
print(f'\n{sep}')
print('CHEST PPG — ALL 4 FILES QUALITY SUMMARY')
print(sep)
print(f"{'Metric':<22} {'V1 Raw':>12} {'V1 Filtered':>14} {'V2 Raw':>12} {'V2 Filtered':>14}")
print('-' * 78)

rows = [
    ('PI % (median)',       lambda d: f"{np.median(d['pis'][d['pis']>0]):.2f}%" if (d['pis']>0).any() else 'N/A'),
    ('SNR dB (median)',     lambda d: f"{np.median(d['snrs']):.1f} dB"),
    ('AC Amp (median)',     lambda d: f"{np.median(d['ac_amps']):.0f} cnts"),
    ('HR valid windows',    lambda d: f"{(((d['hrs']>40)&(d['hrs']<160)).mean()*100):.0f}%"),
    ('RR CV % (median)',    lambda d: f"{np.median(d['rrcvs'][(d['hrs']>40)&(d['hrs']<160)]):.1f}%" if ((d['hrs']>40)&(d['hrs']<160)).any() else 'N/A'),
    ('AGC current',         lambda d: f"{d['agc']:.0f}" if not np.isnan(d['agc']) else 'N/A'),
    ('Duration',            lambda d: f"{d['duration']:.0f}s"),
    ('Sampling rate',       lambda d: f"{d['fs']} Hz"),
]

for label, fn in rows:
    vals = [fn(datasets[n]) for n in ['V1 Raw','V1 Filtered','V2 Raw','V2 Filtered']]
    print(f"  {label:<20} {vals[0]:>12} {vals[1]:>14} {vals[2]:>12} {vals[3]:>14}")

print(sep)
print('\nMETRICS EXPLAINED — What to track and why:')
print('  PI %        -> Primary quality metric. Clinically proven. Must be >0.3% minimum.')
print('  SNR (dB)    -> Frequency-domain quality. >6dB for HR, >10dB for SpO2.')
print('  AC Amplitude-> Raw pulse size. Higher = better skin contact / perfusion.')
print('  HR Valid %  -> How often algorithm detects a plausible heart rate.')
print('                 <50% = signal too noisy for reliable HR extraction.')
print('  RR CV %     -> Beat regularity. <10% = clean. >30% = noisy/false detections.')
print('  AGC Current -> Sensor auto-gain. Low value = good optical contact.')
print('                 High (160 max) = sensor not getting enough reflected light.')
print(sep)
print(f'\nAll plots -> {OUT}')
