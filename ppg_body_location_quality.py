"""
PPG Signal Quality Analysis — AS7058 Multi-Site Dataset
=========================================================
Compares PPG signal quality across three body locations:
  Wrist  (Schema A, 200 Hz, has accelerometer)
  Finger (Schema B, 100 Hz)
  Chest  (Schema B, 100 Hz)

Produces 4 plots in output/ folder:
  01_waveform_comparison.png   — Raw AC waveforms, 15-second window
  02_quality_metrics.png       — PI%, SNR, AC amplitude bar charts
  03_psd_comparison.png        — Power spectral density per dataset
  04_summary_heatmap.png       — Quality heatmap across all datasets

Run:
  py -3 ppg_body_location_quality.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal as sp

# ── File paths ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

DATASETS = [
    # label                site      relative path
    ('Wrist V1',   'Wrist',  'AS7058/01_Wrist_AS7058/V1/wrist_position_nikhil_02.032026.csv'),
    ('Wrist V2',   'Wrist',  'AS7058/01_Wrist_AS7058/v2/wrist_position_nikhil_V2_02.032026.csv'),
    ('Finger V1',  'Finger', 'AS7058/02_Finger_AS7058/V1/Finger_position_nikhil_V1_02.032026_2026-03-02_12-06-03.csv'),
    ('Finger V2',  'Finger', 'AS7058/02_Finger_AS7058/V2/Finger_position_nikhil_V2_02.032026_2026-03-02_12-12-17.csv'),
    ('Finger V3',  'Finger', 'AS7058/04_Finger_AS7058_Parallel with SP-20/Finger_position_nikhil_V3_02.032026_2026-03-02_14-09-26.csv'),
    ('Chest V1',   'Chest',  'AS7058/03_Chest_AS7058/V1_wrist algo/Chest_position_nikhil_V1_02.032026_2026-03-02_13-24-19.csv'),
    ('Chest V2',   'Chest',  'AS7058/03_Chest_AS7058/V2_wrist algo/Chest_position_nikhil_V2_02.032026_2026-03-02_13-36-02.csv'),
]

SITE_COLORS = {'Wrist': '#1565C0', 'Finger': '#2E7D32', 'Chest': '#B71C1C'}

# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataset(filepath, label, site):
    """Load CSV, detect schema, extract clean PPG rows. Returns dict."""
    raw = pd.read_csv(filepath, low_memory=False)

    # Schema A = Wrist (has ACC columns); Schema B = Finger/Chest
    schema = 'A' if 'ACC_X' in raw.columns else 'B'

    # PPG rows: where IR channel (PPG1_SUB1) is not NaN
    raw['_ir']  = pd.to_numeric(raw['PPG1_SUB1'], errors='coerce')
    raw['_red'] = pd.to_numeric(raw['PPG1_SUB2'], errors='coerce')
    raw['_ts']  = pd.to_numeric(raw['TIMESTAMP [s]'], errors='coerce')
    ppg = raw[raw['_ir'].notna()].sort_values('_ts').reset_index(drop=True)

    t   = (ppg['_ts'] - ppg['_ts'].iloc[0]).values
    ir  = ppg['_ir'].values.astype(float)
    red = ppg['_red'].values.astype(float)
    red = np.where(red == 0, np.nan, red)   # mask AGC settling zeros

    fs = int(round(1.0 / pd.Series(t).diff().dropna().median()))

    # AGC settled LED current (first non-NaN AGC1_CURRENT value)
    agc_col = [c for c in raw.columns if 'AGC1' in c and 'CURRENT' in c]
    agc_val = np.nan
    if agc_col:
        agc_series = pd.to_numeric(raw[agc_col[0]], errors='coerce').dropna()
        if len(agc_series):
            agc_val = agc_series.iloc[-1]   # settled value (end of recording)

    # Sensor-reported SpO2 event rows
    sq_col = 'SPO2: SIGNAL_QUALITY'
    events = pd.DataFrame()
    if sq_col in raw.columns:
        raw['_sq'] = pd.to_numeric(raw[sq_col], errors='coerce')
        events = raw[raw['_sq'].notna()].copy()

    return dict(label=label, site=site, schema=schema,
                t=t, ir=ir, red=red, fs=fs, agc=agc_val,
                duration=t[-1], n_rows=len(ppg), events=events)


# ── Signal processing ──────────────────────────────────────────────────────────
def bandpass(sig, fs, low=0.5, high=4.0, order=4):
    nyq = fs / 2.0
    b, a = sp.butter(order, [low / nyq, min(high / nyq, 0.99)], btype='band')
    return sp.filtfilt(b, a, sig)


def compute_quality(ds, window_s=15.0):
    """
    Extract a clean window, compute PI%, SNR, AC amplitude, HR from peaks.
    Returns dict of scalar metrics.
    """
    t, ir, red, fs = ds['t'], ds['ir'], ds['red'], ds['fs']

    # Use t=60–75s window (avoids AGC transients at start)
    start = 60.0
    mask  = (t >= start) & (t <= start + window_s)
    if np.sum(mask) < fs * 5:          # fallback to t=10–25s
        mask = (t >= 10.0) & (t <= 10.0 + window_s)

    t_w  = t[mask]
    ir_w = ir[mask]
    red_w = red[mask]

    # Fill NaN for filtering
    valid_ir  = ~np.isnan(ir_w)
    valid_red = ~np.isnan(red_w)

    ir_fill  = np.where(valid_ir,  ir_w,  np.nanmean(ir_w)  if np.any(valid_ir)  else 0)
    red_fill = np.where(valid_red, red_w, np.nanmean(red_w) if np.any(valid_red) else 0)

    ir_ac  = bandpass(ir_fill,  fs)
    red_ac = bandpass(red_fill, fs)

    ir_ac[~valid_ir]   = np.nan
    red_ac[~valid_red] = np.nan

    # Perfusion Index (PI%) — RED channel, percentile-based
    dc_red = np.nanmean(red_w[valid_red]) if np.any(valid_red) else 1
    ac_ptp = (np.nanpercentile(red_ac[valid_red], 90) -
              np.nanpercentile(red_ac[valid_red], 10)) if np.any(valid_red) else 0
    pi = abs(ac_ptp) / abs(dc_red) * 100 if dc_red != 0 else 0

    # AC amplitude (absolute, ADC counts)
    ac_amp = abs(ac_ptp)

    # SNR via Welch PSD on IR
    ir_hp = np.nan_to_num(ir_ac - np.nanmean(ir_ac))
    f, p  = sp.welch(ir_hp, fs=fs, nperseg=min(len(ir_hp), int(fs * 8)))
    s_pow = np.trapz(p[(f >= 0.7) & (f <= 3.5)], f[(f >= 0.7) & (f <= 3.5)])
    n_pow = np.trapz(p[(f >= 4.0) & (f <= 8.0)], f[(f >= 4.0) & (f <= 8.0)])
    snr   = 10 * np.log10(s_pow / n_pow) if n_pow > 0 else 0

    # HR from peaks on IR AC
    dist  = int(fs * 0.35)
    thr   = np.nanmedian(ir_ac) + 0.4 * np.nanstd(ir_ac)
    peaks, _ = sp.find_peaks(np.nan_to_num(ir_ac), distance=dist, height=thr)
    hr = 0.0
    if len(peaks) > 1:
        rr = np.diff(t_w[peaks])
        hr = 60.0 / np.median(rr)

    # Quality label
    if pi >= 1.0:
        quality = 'GOOD'
    elif pi >= 0.3:
        quality = 'FAIR'
    else:
        quality = 'POOR'

    return dict(pi=pi, snr=snr, ac_amp=ac_amp, dc=dc_red,
                hr=hr, quality=quality,
                t_w=t_w - t_w[0], ir_ac=ir_ac, red_ac=red_ac,
                f=f, p=p, peaks=peaks)


# ── Plot 1: Waveform comparison ────────────────────────────────────────────────
def plot_waveforms(datasets):
    """Stacked waveform panels, one per dataset, same Y scale per site group."""
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        'PPG Waveform Comparison — RED channel, bandpass 0.5-4 Hz, 15-second window\n'
        'Orange triangles = detected heartbeats',
        fontsize=12, fontweight='bold'
    )

    # Shared Y scale per site
    site_amp = {}
    for ds in datasets:
        m = ds['metrics']
        amp = np.nanmax(np.abs(m['red_ac'])) if np.any(~np.isnan(m['red_ac'])) else 1
        site_amp[ds['site']] = max(site_amp.get(ds['site'], 0), amp)

    for ax, ds in zip(axes, datasets):
        m   = ds['metrics']
        c   = SITE_COLORS[ds['site']]
        t   = m['t_w']
        sig = m['red_ac']
        pk  = m['peaks']

        ax.plot(t, sig, color=c, lw=1.0, alpha=0.9)
        ax.fill_between(t, 0, np.where(sig > 0, sig, 0), color=c, alpha=0.12)
        ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

        valid_pk = pk[pk < len(t)]
        if len(valid_pk):
            ax.plot(t[valid_pk], sig[valid_pk], 'v',
                    color='orange', ms=7, zorder=5,
                    label=f'{len(valid_pk)} beats')

        y_lim = site_amp[ds['site']] * 1.3
        ax.set_ylim(-y_lim, y_lim)
        ax.set_ylabel('AC (ADC)', fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

        badge_c = '#2E7D32' if m['quality'] == 'GOOD' else \
                  '#F57F17' if m['quality'] == 'FAIR' else '#B71C1C'
        ax.text(0.995, 0.95,
                f"{m['quality']}  PI={m['pi']:.2f}%  SNR={m['snr']:.1f}dB  HR~{m['hr']:.0f}bpm",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=badge_c, alpha=0.9))

        ax.set_title(f"{ds['label']}  ({ds['site']})",
                     fontsize=9, fontweight='bold', color=c, loc='left')
        if len(valid_pk):
            ax.legend(fontsize=7, loc='upper left')

    axes[-1].set_xlabel('Time (seconds)', fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT, '01_waveform_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


# ── Plot 2: Quality metrics bar charts ────────────────────────────────────────
def plot_quality_metrics(datasets):
    """Bar charts for PI%, SNR, AC amplitude, AGC current side by side."""
    labels  = [ds['label'] for ds in datasets]
    sites   = [ds['site']  for ds in datasets]
    colors  = [SITE_COLORS[s] for s in sites]
    pi_vals  = [ds['metrics']['pi']     for ds in datasets]
    snr_vals = [ds['metrics']['snr']    for ds in datasets]
    amp_vals = [ds['metrics']['ac_amp'] for ds in datasets]
    agc_vals = [ds['agc'] if not np.isnan(ds['agc']) else 0 for ds in datasets]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Signal Quality Metrics by Dataset and Body Location',
                 fontsize=13, fontweight='bold')

    def bar_chart(ax, values, title, ylabel, threshold=None, threshold_label=None):
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='white',
                      linewidth=0.8, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        if threshold is not None:
            ax.axhline(threshold, color='red', lw=1.2, ls='--', alpha=0.7,
                       label=threshold_label)
            ax.legend(fontsize=7)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    bar_chart(axes[0], pi_vals,  'Perfusion Index (PI%)',
              'PI %', threshold=1.0, threshold_label='Threshold 1% (GOOD)')
    bar_chart(axes[1], snr_vals, 'Signal-to-Noise Ratio',
              'SNR (dB)', threshold=6.0, threshold_label='Threshold 6 dB')
    bar_chart(axes[2], [v / 1000 for v in amp_vals],
              'AC Amplitude (RED channel)',
              'AC peak-to-peak (k ADC counts)')
    bar_chart(axes[3], agc_vals, 'AGC LED Current (settled)',
              'AGC1 current (LSB)\nLower = better optical coupling')

    # Site legend
    from matplotlib.patches import Patch
    handles = [Patch(color=SITE_COLORS[s], label=s) for s in ['Wrist', 'Finger', 'Chest']]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUT, '02_quality_metrics.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


# ── Plot 3: PSD comparison ─────────────────────────────────────────────────────
def plot_psd(datasets):
    """Power spectral density for each dataset, grouped by body location."""
    sites  = ['Wrist', 'Finger', 'Chest']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Power Spectral Density — IR channel\n'
                 'The cardiac peak (heartbeat) should appear as a clear spike at 0.8-2.0 Hz',
                 fontsize=12, fontweight='bold')

    for ax, site in zip(axes, sites):
        site_ds = [ds for ds in datasets if ds['site'] == site]
        for ds in site_ds:
            m = ds['metrics']
            ax.semilogy(m['f'], m['p'], lw=1.4, alpha=0.85, label=ds['label'])

        ax.axvspan(0.7, 3.5, alpha=0.08, color='green', label='Cardiac band')
        ax.axvspan(4.0, 8.0, alpha=0.08, color='red',   label='Noise band')
        ax.set_xlim(0, 8)
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('Power (ADC^2/Hz)', fontsize=9)
        ax.set_title(site, fontsize=11, fontweight='bold', color=SITE_COLORS[site])
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

        # Label the bands
        ax.text(2.1,  ax.get_ylim()[0] * 10,  'cardiac\nband',
                fontsize=6, color='green', alpha=0.7, ha='center')
        ax.text(6.0,  ax.get_ylim()[0] * 10,  'noise\nband',
                fontsize=6, color='red',   alpha=0.7, ha='center')

    plt.tight_layout()
    out = os.path.join(OUT, '03_psd_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


# ── Plot 4: Summary heatmap ────────────────────────────────────────────────────
def plot_heatmap(datasets):
    """Colour-coded quality heatmap — rows=datasets, cols=metrics."""
    labels   = [ds['label'] for ds in datasets]
    metrics  = {
        'PI %\n(>1=GOOD)':      [ds['metrics']['pi']            for ds in datasets],
        'SNR dB\n(>6=GOOD)':    [ds['metrics']['snr']           for ds in datasets],
        'AC Amp\n(k counts)':   [ds['metrics']['ac_amp'] / 1000 for ds in datasets],
        'HR\n(BPM)':            [ds['metrics']['hr']            for ds in datasets],
        'AGC\n(lower=better)':  [ds['agc'] if not np.isnan(ds['agc']) else 0
                                 for ds in datasets],
    }

    # Normalize each column 0-1 for colour mapping
    keys = list(metrics.keys())
    data = np.array([metrics[k] for k in keys]).T   # shape (n_datasets, n_metrics)

    # Invert AGC (lower is better)
    agc_col = keys.index('AGC\n(lower=better)')
    col_min = data[:, agc_col].min()
    col_max = data[:, agc_col].max()
    if col_max > col_min:
        data[:, agc_col] = 1 - (data[:, agc_col] - col_min) / (col_max - col_min)

    # Normalize all columns except AGC (already done above)
    data_norm = data.copy().astype(float)
    for c in range(data.shape[1]):
        if c == agc_col:
            continue
        col_min = data[:, c].min()
        col_max = data[:, c].max()
        if col_max > col_min:
            data_norm[:, c] = (data[:, c] - col_min) / (col_max - col_min)
        else:
            data_norm[:, c] = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Site colour band on left
    for i, ds in enumerate(datasets):
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 0.3, 1,
                                   color=SITE_COLORS[ds['site']], clip_on=False))

    # Annotate cells with actual values
    for i in range(len(labels)):
        for j, key in enumerate(keys):
            val = metrics[key][i]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='black' if 0.2 < data_norm[i, j] < 0.8 else 'white')

    ax.set_title('Signal Quality Heatmap — Green = better, Red = worse\n'
                 'Colour side bar: Blue=Wrist, Green=Finger, Red=Chest',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6, label='Relative quality (0=worst, 1=best)')
    plt.tight_layout()
    out = os.path.join(OUT, '04_summary_heatmap.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


# ── Console summary table ──────────────────────────────────────────────────────
def print_summary(datasets):
    sep  = '-' * 82
    hdr  = f"{'Dataset':<14} {'Site':<8} {'Duration':>9} {'Fs':>5} {'PI%':>7} {'SNR dB':>8} {'AC (k)':>8} {'HR BPM':>8} {'AGC':>6} {'Quality':<8}"
    print('\n' + sep)
    print(hdr)
    print(sep)
    for ds in datasets:
        m = ds['metrics']
        print(f"{ds['label']:<14} {ds['site']:<8} "
              f"{ds['duration']:>8.0f}s "
              f"{ds['fs']:>5} Hz "
              f"{m['pi']:>6.2f}% "
              f"{m['snr']:>7.1f}  "
              f"{m['ac_amp']/1000:>7.1f}k "
              f"{m['hr']:>7.1f}  "
              f"{ds['agc']:>5.0f}  "
              f"{m['quality']}")
    print(sep)
    print('\nQuality thresholds:  PI% >= 1.0 -> GOOD  |  PI% >= 0.3 -> FAIR  |  PI% < 0.3 -> POOR')
    print('                     SNR >= 6 dB recommended for HR/SpO2 algorithm development\n')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('Loading datasets ...')
    datasets = []
    for label, site, relpath in DATASETS:
        path = os.path.join(BASE, relpath)
        if not os.path.exists(path):
            print(f'  [SKIP] {label} — file not found: {relpath}')
            continue
        ds = load_dataset(path, label, site)
        ds['metrics'] = compute_quality(ds)
        print(f'  {label:<14} PI={ds["metrics"]["pi"]:.2f}%  '
              f'SNR={ds["metrics"]["snr"]:.1f}dB  '
              f'HR~{ds["metrics"]["hr"]:.0f}bpm  '
              f'AGC={ds["agc"]:.0f}  -> {ds["metrics"]["quality"]}')
        datasets.append(ds)

    print('\nGenerating plots ...')
    plot_waveforms(datasets)
    plot_quality_metrics(datasets)
    plot_psd(datasets)
    plot_heatmap(datasets)

    print_summary(datasets)
    print(f'All outputs saved to: {OUT}')


if __name__ == '__main__':
    main()
