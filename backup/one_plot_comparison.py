"""
One plot: Finger vs Wrist vs Chest signal quality comparison.
Shows 15 seconds of the RED channel heartbeat waveform for each body location.
"""

import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    'Finger': 'AS7058/02_Finger_AS7058/V1/Finger_position_nikhil_V1_02.032026_2026-03-02_12-06-03.csv',
    'Wrist':  'AS7058/01_Wrist_AS7058/V1/wrist_position_nikhil_02.032026.csv',
    'Chest':  'AS7058/03_Chest_AS7058/V1_wrist algo/Chest_position_nikhil_V1_02.032026_2026-03-02_13-24-19.csv',
}

COLORS = {'Finger': '#2E7D32', 'Wrist': '#1565C0', 'Chest': '#B71C1C'}

def load_red(path):
    raw = pd.read_csv(path, low_memory=False)
    raw['ir']  = pd.to_numeric(raw['PPG1_SUB1'], errors='coerce')
    raw['red'] = pd.to_numeric(raw['PPG1_SUB2'], errors='coerce').fillna(0)
    raw['ts']  = pd.to_numeric(raw['TIMESTAMP [s]'], errors='coerce')
    ppg = raw[raw['ir'].notna()].sort_values('ts').reset_index(drop=True)
    t   = (ppg['ts'] - ppg['ts'].iloc[0]).values
    red = ppg['red'].values.astype(float)
    red = np.where(red == 0, np.nan, red)
    fs  = round(1.0 / pd.Series(t).diff().dropna().median())
    return t, red, int(fs)

def bandpass(sig, fs):
    b, a = sp.butter(4, [0.5/(fs/2), min(4.0/(fs/2), 0.99)], btype='band')
    return sp.filtfilt(b, a, sig)

def perfusion_index(raw_sig, bp_sig):
    dc = np.nanmean(raw_sig)
    ac = np.ptp(bp_sig)
    return (ac / dc * 100) if dc > 0 else 0

# ── Load + process ────────────────────────────────────────────────────────────
print("Loading data ...")
data = {}
for site, relpath in FILES.items():
    t, red, fs = load_red(os.path.join(BASE, relpath))

    # Pick a clean 15-second window well into the recording (avoids AGC transients)
    # Use t=80-95s for Wrist (shorter recording), t=100-115s for Finger/Chest
    start_s = 60.0
    end_s   = start_s + 15.0
    mask = (t >= start_s) & (t <= end_s)
    # Fallback if not enough data
    if np.sum(mask) < int(fs * 5):
        start_s, end_s = 10.0, 25.0
        mask = (t >= start_s) & (t <= end_s)

    t_w   = t[mask] - t[mask][0]          # reset to 0
    red_w = red[mask]

    # Fill NaN for filtering
    valid = ~np.isnan(red_w)
    fill  = np.where(valid, red_w, np.nanmean(red_w[valid]) if np.any(valid) else 0)
    red_ac = bandpass(fill, fs)
    red_ac[~valid] = np.nan

    # Detect heartbeat peaks — use median-based threshold (robust against spikes)
    dist = int(fs * 0.35)                 # min 0.35s between peaks (~170 BPM max)
    med  = np.nanmedian(red_ac)
    std  = np.nanstd(red_ac)
    thr  = med + 0.4 * std                # threshold above median
    peaks, _ = sp.find_peaks(red_ac, distance=dist, height=thr)

    # Perfusion Index — use 90th-10th percentile range (robust against outliers)
    dc   = np.nanmean(red_w[valid]) if np.any(valid) else 1
    ac   = np.percentile(red_ac[valid], 90) - np.percentile(red_ac[valid], 10) if np.any(valid) else 0
    pi   = abs(ac) / abs(dc) * 100 if dc != 0 else 0

    # SNR: cardiac band power vs noise power
    try:
        red_hp = red_ac - np.nanmean(red_ac)
        red_hp = np.nan_to_num(red_hp)
        f, p   = sp.welch(red_hp, fs=fs, nperseg=min(len(red_hp), int(fs*8)))
        s_pow  = np.trapz(p[(f>=0.7)&(f<=3.5)], f[(f>=0.7)&(f<=3.5)])
        n_pow  = np.trapz(p[(f>=4.0)&(f<=8.0)], f[(f>=4.0)&(f<=8.0)])
        snr    = 10*np.log10(s_pow/n_pow) if n_pow > 0 else 0
    except:
        snr = 0

    data[site] = dict(t=t_w, red_ac=red_ac, peaks=peaks, pi=pi, snr=snr, fs=fs)
    print(f"  {site}: PI={pi:.2f}%  SNR={snr:.1f}dB  peaks={len(peaks)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

# Compute shared Y-axis range so all 3 are on same scale
all_amps = [np.nanmax(np.abs(data[s]['red_ac'])) for s in ['Finger','Wrist','Chest']]
y_max = max(all_amps) * 1.25

fig.suptitle(
    'PPG Signal Quality Comparison — Finger vs Wrist vs Chest\n'
    'RED channel (~660 nm), bandpass filtered 0.5–4 Hz, same Y-axis scale\n'
    'Each orange triangle = one detected heartbeat',
    fontsize=12, fontweight='bold'
)

for ax, site in zip(axes, ['Finger', 'Wrist', 'Chest']):
    d   = data[site]
    c   = COLORS[site]
    t   = d['t']
    sig = d['red_ac']
    pk  = d['peaks']

    # Waveform
    ax.plot(t, sig, color=c, lw=1.2, alpha=0.9)

    # Heartbeat peaks
    valid_pk = pk[pk < len(t)]
    if len(valid_pk):
        ax.plot(t[valid_pk], sig[valid_pk], 'v',
                color='orange', ms=8, zorder=5, label=f'{len(valid_pk)} beats detected')

    # Zero line
    ax.axhline(0, color='gray', lw=0.6, linestyle='--', alpha=0.5)

    # Shade area under positive half (pulsatile component)
    ax.fill_between(t, 0, np.where(sig > 0, sig, 0), color=c, alpha=0.15)

    # Quality badge in top-right
    quality_label = 'GOOD'   if d['pi'] >= 1.0 else \
                    'FAIR'   if d['pi'] >= 0.3 else 'POOR'
    badge_color   = '#2E7D32' if quality_label == 'GOOD' else \
                    '#F57F17' if quality_label == 'FAIR' else '#B71C1C'

    ax.text(0.99, 0.95,
            f"{quality_label}\nPI = {d['pi']:.2f}%\nSNR = {d['snr']:.1f} dB",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=badge_color, alpha=0.9))

    # Y-axis: same scale for all
    ax.set_ylim(-y_max, y_max)
    ax.set_ylabel('Signal Amplitude\n(ADC counts, AC)', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

    # Title with plain-language explanation
    n_beats = len(valid_pk)
    hr_est  = 60.0 / np.median(np.diff(t[valid_pk])) if n_beats > 1 else 0
    hr_str  = f'~{hr_est:.0f} BPM' if hr_est > 0 else 'HR undetectable'

    desc = {
        'Finger': 'High blood flow, direct arterial pulsation → strongest signal',
        'Wrist':  'Moderate blood flow, some motion artifact → usable signal',
        'Chest':  'Low surface blood flow, weak pulsation → too weak for SpO2',
    }
    ax.set_title(
        f'{site}  |  {hr_str}  |  {desc[site]}',
        fontsize=10, fontweight='bold', color=c, loc='left'
    )
    if n_beats > 1:
        ax.legend(fontsize=8, loc='upper left')

axes[-1].set_xlabel('Time (seconds)  —  15-second window starting at t=30s', fontsize=10)

# Annotation explaining what to look for
fig.text(0.01, 0.01,
    'How to read this chart:\n'
    '  • Tall, regular waves = good heartbeat signal\n'
    '  • Flat or noisy line = poor signal quality\n'
    '  • All 3 panels use the SAME vertical scale for fair comparison\n'
    '  • PI% = pulsatile amplitude / baseline × 100  (clinical threshold: >1% = good, >0.3% = acceptable)',
    fontsize=8, color='#333333',
    bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 1])

out = os.path.join(BASE, 'output_sq', 'SIGNAL_QUALITY_COMPARISON.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f"\nSaved -> {out}")
