"""
Finger vs Wrist vs Chest — PPG Quality Comparison
===================================================
Easy-to-understand side-by-side quality analysis.
Uses the best representative file from each body location.

Output: output/comparison/
  01_heartbeat_waveforms.png   → Can you see a heartbeat? Side by side
  02_metrics_comparison.png    → 6 metrics, traffic-light colour coded
  03_verdict_report.png        → Plain-language summary card

Run: py -3 compare_locations.py
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
OUT  = os.path.join(BASE, 'output', 'comparison')
os.makedirs(OUT, exist_ok=True)

# ── Best representative file from each location ───────────────────────────────
LOCATIONS = {
    'Finger': {
        'path':  'AS7058/04_Finger_AS7058_Parallel with SP-20/Finger_position_nikhil_V3_02.032026_2026-03-02_14-09-26.csv',
        'color': '#2E7D32',
        'icon':  'Finger\n(V3, best)',
        'desc':  'Fingertip — high blood flow,\ndirect arterial pulse',
    },
    'Wrist': {
        'path':  'AS7058/01_Wrist_AS7058/V1/wrist_position_nikhil_02.032026.csv',
        'color': '#1565C0',
        'icon':  'Wrist\n(V1)',
        'desc':  'Wrist — moderate blood flow,\nsome motion artifact',
    },
    'Chest': {
        'path':  'AS7058/03_Chest_AS7058/V2_wrist algo/Chest_position_nikhil_V2_02.032026_2026-03-02_13-36-02.csv',
        'color': '#B71C1C',
        'icon':  'Chest\n(V2, best)',
        'desc':  'Chest — low surface blood flow,\nweak pulsation',
    },
}

# Quality thresholds
THR = dict(pi_good=1.0, pi_fair=0.3, snr_good=10.0, snr_fair=6.0,
           rrcv_good=10.0, rrcv_fair=25.0, hr_lo=40.0, hr_hi=160.0)

# ── Load ──────────────────────────────────────────────────────────────────────
def load(path):
    raw = pd.read_csv(path, low_memory=False)
    s1  = pd.to_numeric(raw['PPG1_SUB1'], errors='coerce')
    ts  = pd.to_numeric(raw['TIMESTAMP [s]'], errors='coerce')

    # PPG rows (large DC for raw files)
    mask = s1.notna() & (s1 > 1000)
    ppg  = raw[mask].copy()
    t    = pd.to_numeric(ppg['TIMESTAMP [s]'], errors='coerce').values
    ir   = pd.to_numeric(ppg['PPG1_SUB1'],     errors='coerce').values
    red  = pd.to_numeric(ppg['PPG1_SUB2'],     errors='coerce').values

    idx = np.argsort(t)
    t, ir, red = t[idx], ir[idx], red[idx]
    t = t - t[0]

    dt = np.diff(t)
    fs = int(round(1.0 / np.median(dt[dt > 0])))

    # RED: mask AGC settling zeros
    red = np.where(red == 0, np.nan, red.astype(float))

    # AGC
    agc_col = [c for c in raw.columns if 'AGC1' in c and 'CURRENT' in c]
    agc = float(pd.to_numeric(raw[agc_col[0]], errors='coerce').dropna().iloc[-1]) \
          if agc_col else np.nan

    # Sensor events
    sq_col = 'SPO2: SIGNAL_QUALITY'
    ev_t, ev_pi, ev_hr = np.array([]), np.array([]), np.array([])
    if sq_col in raw.columns:
        sq_mask = pd.to_numeric(raw[sq_col], errors='coerce').notna()
        ev = raw[sq_mask]
        ev_t  = pd.to_numeric(ev['TIMESTAMP [s]'], errors='coerce').values
        ev_t  = ev_t - ev_t[0] if len(ev_t) else ev_t
        ev_pi = pd.to_numeric(ev.get('SPO2: PI [%]', pd.Series()), errors='coerce').values
        ev_hr = pd.to_numeric(ev.get('SPO2: HEART_RATE [bpm]', pd.Series()), errors='coerce').values

    return dict(t=t, ir=ir, red=red, fs=fs, agc=agc,
                ev_t=ev_t, ev_pi=ev_pi, ev_hr=ev_hr,
                duration=float(t[-1]))


def bandpass(sig, fs, lo=0.5, hi=4.0):
    nyq = fs / 2.0
    b, a = sp.butter(4, [lo/nyq, min(hi/nyq, 0.99)], btype='band')
    return sp.filtfilt(b, a, sig)


# ── Sliding metrics (10-second windows) ───────────────────────────────────────
def sliding_metrics(ds, win=10.0, step=2.0):
    t, ir, red, fs = ds['t'], ds['ir'], ds['red'], ds['fs']
    wt, pis, snrs, amps, hrs, rvs = [], [], [], [], [], []

    start = 10.0   # skip first 10s for AGC settling
    while start + win <= ds['duration']:
        end  = start + win
        m    = (t >= start) & (t < end)
        seg_ir  = ir[m].copy()
        seg_red = red[m].copy()
        ts_s = t[m]
        if len(seg_ir) < fs * 3:
            start += step; continue

        seg_ir = np.clip(seg_ir, np.percentile(seg_ir, 1), np.percentile(seg_ir, 99))
        ac = bandpass(seg_ir - np.mean(seg_ir), fs)   # IR for HR/SNR

        # PI uses RED channel (much larger AC amplitude, clinical standard)
        valid_red = ~np.isnan(seg_red)
        if np.sum(valid_red) > fs * 2:
            red_fill = np.where(valid_red, seg_red, np.nanmean(seg_red[valid_red]))
            red_ac   = bandpass(red_fill - np.nanmean(red_fill), fs)
            dc_red   = np.nanmean(seg_red[valid_red])
            ac_red   = np.percentile(red_ac[valid_red], 90) - np.percentile(red_ac[valid_red], 10)
            pi       = abs(ac_red) / abs(dc_red) * 100 if dc_red > 0 else 0
        else:
            dc  = np.mean(seg_ir)
            ac_ptp_ir = np.percentile(ac, 90) - np.percentile(ac, 10)
            pi  = abs(ac_ptp_ir) / dc * 100 if dc > 0 else 0

        ac_ptp = np.percentile(ac, 90) - np.percentile(ac, 10)

        f, p = sp.welch(np.nan_to_num(ac - np.mean(ac)), fs=fs,
                        nperseg=min(len(ac), int(fs*8)))
        sp_  = np.trapz(p[(f>=0.7)&(f<=3.5)], f[(f>=0.7)&(f<=3.5)])
        np_  = np.trapz(p[(f>=4.0)&(f<=8.0)], f[(f>=4.0)&(f<=8.0)])
        snr  = 10*np.log10(sp_/np_) if sp_>0 and np_>0 else 0

        dist = int(fs * 0.35)
        thr  = np.median(ac) + 0.35*np.std(ac)
        peaks, _ = sp.find_peaks(ac, distance=dist, height=thr)
        hr, rrcv = 0.0, 99.0
        if len(peaks) >= 2:
            rr = np.diff(ts_s[peaks])
            rr = rr[(rr > 0.3) & (rr < 2.0)]
            if len(rr):
                hr   = 60.0 / np.median(rr)
                rrcv = np.std(rr) / np.mean(rr) * 100

        wt.append(start + win/2)
        pis.append(pi); snrs.append(snr); amps.append(abs(ac_ptp))
        hrs.append(hr); rvs.append(rrcv)
        start += step

    return dict(wt=np.array(wt), pis=np.array(pis), snrs=np.array(snrs),
                amps=np.array(amps), hrs=np.array(hrs), rvs=np.array(rvs))


def verdict(val, metric):
    if metric == 'pi':
        if val >= THR['pi_good']:  return 'GOOD', '#2E7D32'
        if val >= THR['pi_fair']:  return 'FAIR', '#F57F17'
        return 'POOR', '#C62828'
    if metric == 'snr':
        if val >= THR['snr_good']: return 'GOOD', '#2E7D32'
        if val >= THR['snr_fair']: return 'FAIR', '#F57F17'
        return 'POOR', '#C62828'
    if metric == 'rrcv':
        if val <= THR['rrcv_good']: return 'GOOD', '#2E7D32'
        if val <= THR['rrcv_fair']: return 'FAIR', '#F57F17'
        return 'POOR', '#C62828'
    if metric == 'hrpct':
        if val >= 80: return 'GOOD', '#2E7D32'
        if val >= 50: return 'FAIR', '#F57F17'
        return 'POOR', '#C62828'
    if metric == 'agc':
        if val <= 30:  return 'GOOD', '#2E7D32'
        if val <= 80:  return 'FAIR', '#F57F17'
        return 'POOR', '#C62828'
    return 'N/A', '#888888'


# ── Load all locations ────────────────────────────────────────────────────────
print('Loading data ...')
data = {}
for loc, cfg in LOCATIONS.items():
    ds = load(os.path.join(BASE, cfg['path']))
    ds['metrics'] = sliding_metrics(ds)
    ds.update(cfg)
    data[loc] = ds
    m   = ds['metrics']
    vhr = (m['hrs'] > THR['hr_lo']) & (m['hrs'] < THR['hr_hi'])
    print(f"  {loc:<8} PI={np.median(m['pis']):.2f}%  SNR={np.median(m['snrs']):.1f}dB  "
          f"AC={np.median(m['amps']):.0f}  HR_valid={vhr.mean()*100:.0f}%  AGC={ds['agc']:.0f}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Heartbeat waveforms side by side
# ════════════════════════════════════════════════════════════════════════════
print('\nPlot 1: Waveforms ...')

WIN_START = 60.0
WIN_LEN   = 15.0

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#FAFAFA')
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)
axes = [fig.add_subplot(gs[i]) for i in range(3)]

fig.suptitle(
    'Can you see a heartbeat? — 15-second PPG signal from each body location\n'
    'RED channel, bandpass filtered 0.5-4 Hz  |  Triangles = detected heartbeats',
    fontsize=14, fontweight='bold', y=0.98
)

# Shared amplitude scale across all 3
all_max = []
for loc in ['Finger', 'Wrist', 'Chest']:
    ds = data[loc]
    m  = (ds['t'] >= WIN_START) & (ds['t'] < WIN_START + WIN_LEN)
    seg = ds['ir'][m]
    if len(seg) > ds['fs']:
        ac = bandpass(seg - np.mean(seg), ds['fs'])
        all_max.append(np.percentile(np.abs(ac), 98))
shared_ylim = max(all_max) * 1.35 if all_max else 1000

for ax, loc in zip(axes, ['Finger', 'Wrist', 'Chest']):
    ds  = data[loc]
    c   = ds['color']
    t   = ds['t']
    ir  = ds['ir']
    fs  = ds['fs']
    m_q = ds['metrics']

    mask = (t >= WIN_START) & (t < WIN_START + WIN_LEN)
    seg  = ir[mask]
    ts_w = t[mask] - t[mask][0]

    if len(seg) < fs * 3:
        ax.text(0.5, 0.5, 'Not enough data in this window',
                transform=ax.transAxes, ha='center', va='center')
        continue

    seg_c = np.clip(seg, np.percentile(seg, 0.5), np.percentile(seg, 99.5))
    ac    = bandpass(seg_c - np.mean(seg_c), fs)

    # Waveform
    ax.plot(ts_w, ac, color=c, lw=1.2, alpha=0.92)
    ax.fill_between(ts_w, 0, np.where(ac > 0, ac, 0), color=c, alpha=0.15)
    ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.4)

    # Peak detection
    dist = int(fs * 0.35)
    thr  = np.median(ac) + 0.4 * np.std(ac)
    peaks, _ = sp.find_peaks(ac, distance=dist, height=thr)
    hr_est = 0
    if len(peaks) >= 2:
        rr = np.diff(ts_w[peaks])
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr):
            hr_est = 60.0 / np.median(rr)
    if len(peaks):
        ax.plot(ts_w[peaks], ac[peaks], 'v', color='#FF8F00',
                ms=10, zorder=6, label=f'{len(peaks)} beats  ~{hr_est:.0f} BPM')

    # Y axis same scale
    ax.set_ylim(-shared_ylim, shared_ylim)
    ax.set_ylabel('Signal\n(ADC counts)', fontsize=9)
    ax.grid(True, alpha=0.18, zorder=0)
    ax.tick_params(labelsize=8)

    # Quality badge
    pi_med  = np.nanmedian(m_q['pis'])
    snr_med = np.nanmedian(m_q['snrs'])
    lbl_pi, col_pi   = verdict(pi_med, 'pi')
    badge_col = col_pi
    ax.text(0.995, 0.96,
            f"PI = {pi_med:.2f}%  [{lbl_pi}]\nSNR = {snr_med:.1f} dB",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=badge_col, alpha=0.92))

    # Title with plain description
    ax.set_title(
        f'{loc}   |   {ds["desc"]}',
        fontsize=11, fontweight='bold', color=c, loc='left', pad=4
    )
    if len(peaks):
        ax.legend(loc='upper left', fontsize=8, framealpha=0.7)

axes[-1].set_xlabel('Time (seconds)', fontsize=10)

# Annotation box at bottom
fig.text(0.01, 0.01,
    'How to read this chart:   '
    '  Tall, regular waves = good heartbeat signal   '
    '  Flat or noisy line = poor signal quality   '
    '  All 3 panels use the SAME vertical scale for fair comparison',
    fontsize=8, color='#444',
    bbox=dict(boxstyle='round', facecolor='#eeeeee', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
p1 = os.path.join(OUT, '01_heartbeat_waveforms.png')
fig.savefig(p1, dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print(f'  Saved -> {p1}')


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Metrics comparison: traffic-light table
# ════════════════════════════════════════════════════════════════════════════
print('Plot 2: Metrics comparison ...')

METRICS_DEF = [
    ('PI %',         'pi',    'How strong\nis the pulse?',
     'AC heartbeat size / DC baseline\n> 1% = GOOD   > 0.3% = OK   < 0.3% = POOR'),
    ('SNR (dB)',      'snr',   'How clean\nis the signal?',
     'Heartbeat frequency power vs noise\n> 10 dB = GOOD   > 6 dB = OK   < 6 dB = POOR'),
    ('AC Amplitude',  'amp',   'How big is\nthe heartbeat?',
     'Peak-to-peak size of heartbeat ripple\nHigher = stronger pulse — no fixed threshold'),
    ('HR Valid %',    'hrpct', 'How reliably\ndetected HR?',
     'Windows where HR was 40-160 BPM\n> 80% = GOOD   > 50% = OK   < 50% = POOR'),
    ('RR CV %',       'rrcv',  'How regular\nare the beats?',
     'Beat-to-beat interval variation\n< 10% = GOOD   < 25% = OK   > 25% = POOR'),
    ('AGC Current',   'agc',   'Sensor contact\nwith skin?',
     'LED drive current (lower = better coupling)\n< 30 = GOOD   < 80 = OK   > 80 = POOR'),
]

fig, axes = plt.subplots(len(METRICS_DEF), 3,
                          figsize=(15, 2.6 * len(METRICS_DEF)))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Signal Quality Comparison — Finger vs Wrist vs Chest\n'
             'Green = GOOD  |  Orange = FAIR  |  Red = POOR',
             fontsize=14, fontweight='bold')

# Column headers
for col, loc in enumerate(['Finger', 'Wrist', 'Chest']):
    axes[0][col].set_title(f'{LOCATIONS[loc]["icon"]}',
                           fontsize=13, fontweight='bold',
                           color=LOCATIONS[loc]['color'],
                           bbox=dict(boxstyle='round,pad=0.4',
                                     facecolor=LOCATIONS[loc]['color'],
                                     alpha=0.12))

for row, (name, key, short_q, explanation) in enumerate(METRICS_DEF):
    ds_vals = {}
    for loc in ['Finger', 'Wrist', 'Chest']:
        m   = data[loc]['metrics']
        vhr = (m['hrs'] > THR['hr_lo']) & (m['hrs'] < THR['hr_hi'])
        if key == 'pi':
            val = np.nanmedian(m['pis'])
        elif key == 'snr':
            val = np.nanmedian(m['snrs'])
        elif key == 'amp':
            val = np.nanmedian(m['amps'])
        elif key == 'hrpct':
            val = vhr.mean() * 100
        elif key == 'rrcv':
            val = np.nanmedian(m['rvs'][vhr]) if vhr.any() else 99.0
        elif key == 'agc':
            val = data[loc]['agc']
        ds_vals[loc] = val

    for col, loc in enumerate(['Finger', 'Wrist', 'Chest']):
        ax  = axes[row][col]
        val = ds_vals[loc]
        ax.set_facecolor('#F8F9FA')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if np.isnan(val):
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='gray')
            continue

        lbl, col_c = verdict(val, key)

        # Coloured tile background
        ax.set_facecolor(col_c + '22')  # very light tint
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(col_c)
            spine.set_linewidth(2.5)

        # Format value
        if key == 'pi':
            val_str = f'{val:.3f} %'
        elif key == 'snr':
            val_str = f'{val:.1f} dB'
        elif key == 'amp':
            val_str = f'{val:.0f} counts'
        elif key == 'hrpct':
            val_str = f'{val:.0f} %'
        elif key == 'rrcv':
            val_str = f'{val:.1f} %'
        elif key == 'agc':
            val_str = f'{val:.0f} LSB'

        # Large value
        ax.text(0.5, 0.62, val_str, transform=ax.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=col_c)
        # Badge
        ax.text(0.5, 0.22, lbl, transform=ax.transAxes,
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=col_c, alpha=0.95))

    # Row label on the left
    ax0 = axes[row][0]
    ax0.set_ylabel(f'{name}\n{short_q}', fontsize=9, fontweight='bold',
                   rotation=0, labelpad=120, va='center')

# Explanation row at bottom
for row, (name, key, short_q, explanation) in enumerate(METRICS_DEF):
    axes[row][2].text(1.02, 0.5, explanation,
                      transform=axes[row][2].transAxes,
                      fontsize=7.5, va='center', color='#444',
                      bbox=dict(boxstyle='round', facecolor='#eeeeee', alpha=0.7))

plt.tight_layout(rect=[0.08, 0, 0.88, 0.96])
p2 = os.path.join(OUT, '02_metrics_comparison.png')
fig.savefig(p2, dpi=160, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f'  Saved -> {p2}')


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Verdict report card
# ════════════════════════════════════════════════════════════════════════════
print('Plot 3: Verdict report ...')

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle('PPG Data Quality — Validation Report\nAS7058 Sensor, 02.03.2026',
             fontsize=14, fontweight='bold')

VERDICTS = {
    'Finger': {
        'overall':  'GOOD',
        'color':    '#2E7D32',
        'score':    '4 / 5',
        'usable_for': ['Heart Rate (HR)', 'SpO2 (with calibration)',
                       'HRV metrics', 'Respiratory Rate'],
        'not_for':  ['None — best location'],
        'notes':    'Strongest signal. Direct arterial\npulsation. Recommended for\nalgorithm development.',
    },
    'Wrist': {
        'overall':  'FAIR',
        'color':    '#F57F17',
        'score':    '3 / 5',
        'usable_for': ['Heart Rate (HR)', 'HRV (with filtering)',
                       'Step count / Activity'],
        'not_for':  ['SpO2 (signal too weak)', 'Clinical-grade accuracy'],
        'notes':    'Usable but needs motion\nartifact filtering.\nSuitable for wellness use.',
    },
    'Chest': {
        'overall':  'POOR',
        'color':    '#C62828',
        'score':    '1 / 5',
        'usable_for': ['Experimental HR only'],
        'not_for':  ['SpO2', 'HRV', 'Any clinical use'],
        'notes':    'Pulsatile signal too weak\n(PI = 0.002%).\nNeeds algorithm redesign\nor different sensor position.',
    },
}

# Fill colours
OVERALL_BG = {'GOOD': '#E8F5E9', 'FAIR': '#FFF8E1', 'POOR': '#FFEBEE'}

for ax, loc in zip(axes, ['Finger', 'Wrist', 'Chest']):
    v   = VERDICTS[loc]
    c   = v['color']
    ax.set_facecolor(OVERALL_BG[v['overall']])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(c); spine.set_linewidth(3)

    # Header
    ax.text(5, 9.3, loc, ha='center', va='center', fontsize=18,
            fontweight='bold', color=c)

    # Overall badge
    ax.text(5, 8.1, v['overall'], ha='center', va='center', fontsize=22,
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=c, alpha=0.95))

    # Score
    ax.text(5, 7.1, f'Score: {v["score"]}', ha='center', va='center',
            fontsize=13, color=c, fontweight='bold')

    # Metrics summary bar (6 dots)
    m   = data[loc]['metrics']
    vhr = (m['hrs'] > THR['hr_lo']) & (m['hrs'] < THR['hr_hi'])
    checks = [
        verdict(np.nanmedian(m['pis']),     'pi')[0],
        verdict(np.nanmedian(m['snrs']),    'snr')[0],
        'N/A',
        verdict(vhr.mean()*100,             'hrpct')[0],
        verdict(np.nanmedian(m['rvs'][vhr]) if vhr.any() else 99, 'rrcv')[0],
        verdict(data[loc]['agc'],           'agc')[0],
    ]
    metric_labels = ['PI%', 'SNR', 'AC', 'HR%', 'RR', 'AGC']
    dot_colors = {'GOOD': '#2E7D32', 'FAIR': '#F57F17', 'POOR': '#C62828', 'N/A': '#9E9E9E'}
    for i, (chk, mlbl) in enumerate(zip(checks, metric_labels)):
        x = 1.0 + i * 1.5
        ax.plot(x, 6.1, 'o', color=dot_colors[chk], ms=14, zorder=5)
        if chk == 'GOOD':
            ax.text(x, 6.1, '+', ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')
        elif chk == 'POOR':
            ax.text(x, 6.1, 'x', ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold')
        ax.text(x, 5.55, mlbl, ha='center', va='center', fontsize=7, color='#555')

    # Usable for
    ax.text(0.4, 5.0, 'Usable for:', fontsize=9, fontweight='bold', color='#2E7D32', va='top')
    for i, item in enumerate(v['usable_for']):
        ax.text(0.4, 4.4 - i*0.55, f'  + {item}', fontsize=8.5, color='#2E7D32', va='top')

    # Not for
    y_start = 4.4 - len(v['usable_for'])*0.55 - 0.2
    ax.text(0.4, y_start, 'Not suitable for:', fontsize=9, fontweight='bold',
            color='#C62828', va='top')
    for i, item in enumerate(v['not_for']):
        ax.text(0.4, y_start - 0.55 - i*0.55, f'  - {item}', fontsize=8.5,
                color='#C62828', va='top')

    # Notes
    ax.text(5, 0.5, v['notes'], ha='center', va='bottom', fontsize=8.5,
            color='#444', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

plt.tight_layout(rect=[0, 0, 1, 0.94])
p3 = os.path.join(OUT, '03_verdict_report.png')
fig.savefig(p3, dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print(f'  Saved -> {p3}')


# ── Console summary ───────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('VALIDATION SUMMARY — Finger vs Wrist vs Chest')
print('=' * 60)
rows = []
for loc in ['Finger', 'Wrist', 'Chest']:
    m   = data[loc]['metrics']
    vhr = (m['hrs'] > THR['hr_lo']) & (m['hrs'] < THR['hr_hi'])
    pi  = np.nanmedian(m['pis'])
    snr = np.nanmedian(m['snrs'])
    amp = np.nanmedian(m['amps'])
    hrp = vhr.mean() * 100
    rv  = np.nanmedian(m['rvs'][vhr]) if vhr.any() else 99.0
    agc = data[loc]['agc']
    v_pi, _  = verdict(pi,  'pi')
    v_snr, _ = verdict(snr, 'snr')
    v_hr, _  = verdict(hrp, 'hrpct')
    v_rv, _  = verdict(rv,  'rrcv')
    v_agc, _ = verdict(agc, 'agc')
    overall  = VERDICTS[loc]['overall']
    print(f'\n  {loc} [{overall}]')
    print(f'    PI%       : {pi:.3f}%   [{v_pi}]')
    print(f'    SNR       : {snr:.1f} dB  [{v_snr}]')
    print(f'    AC Amp    : {amp:.0f} counts')
    print(f'    HR Valid  : {hrp:.0f}%     [{v_hr}]')
    print(f'    RR CV     : {rv:.1f}%    [{v_rv}]')
    print(f'    AGC       : {agc:.0f} LSB   [{v_agc}]')
print('\n' + '=' * 60)
print(f'\nAll plots saved to: {OUT}')
