"""
PPG Signal Quality Analysis — AS7058 Multi-Site
================================================
Focused entirely on signal quality metrics. No algorithm output.

Plots generated in output_sq/ :
  01_pi_over_time.png          Perfusion Index (%) per dataset over time
  02_ac_amplitude_over_time.png  AC peak-to-peak amplitude (RED & IR) over time
  03_dc_baseline_over_time.png   DC baseline drift (slow trend) per channel
  04_snr_over_time.png           Sliding-window SNR (dB) per dataset
  05_spectrogram.png             Time-frequency spectrogram (RED channel)
  06_psd_per_dataset.png         Power spectral density — all datasets overlaid per site
  07_beat_template.png           Averaged heartbeat waveform (beat template)
  08_ir_red_correlation.png      IR vs RED amplitude cross-plot + rolling correlation
  09_sensor_events.png           Sensor-reported PI, R-ratio, SpO2, HR (sparse events)
  10_agc_settling.png            AGC LED current vs time (how sensor auto-adjusted power)
  11_quality_heatmap.png         All quality metrics in one colour-coded heatmap
  12_waveform_overlay_30s.png    AC-filtered RED signal for all datasets on same axes

Axis convention used everywhere:
  X = Time (seconds elapsed from recording start)
  Y = metric-specific (labelled on each plot)
"""

import os, sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy import signal as sp
from scipy.signal import hilbert

warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'output_sq')

DATASETS = {
    'Wrist V1':  'AS7058/01_Wrist_AS7058/V1/wrist_position_nikhil_02.032026.csv',
    'Wrist V2':  'AS7058/01_Wrist_AS7058/v2/wrist_position_nikhil_V2_02.032026.csv',
    'Finger V1': 'AS7058/02_Finger_AS7058/V1/Finger_position_nikhil_V1_02.032026_2026-03-02_12-06-03.csv',
    'Finger V2': 'AS7058/02_Finger_AS7058/V2/Finger_position_nikhil_V2_02.032026_2026-03-02_12-12-17.csv',
    'Finger V3': 'AS7058/04_Finger_AS7058_Parallel with SP-20/Finger_position_nikhil_V3_02.032026_2026-03-02_14-09-26.csv',
    'Chest V1':  'AS7058/03_Chest_AS7058/V1_wrist algo/Chest_position_nikhil_V1_02.032026_2026-03-02_13-24-19.csv',
    'Chest V2':  'AS7058/03_Chest_AS7058/V2_wrist algo/Chest_position_nikhil_V2_02.032026_2026-03-02_13-36-02.csv',
}

SITE_COLOR = {
    'Wrist':  {'V1': '#1565C0', 'V2': '#64B5F6'},
    'Finger': {'V1': '#2E7D32', 'V2': '#81C784', 'V3': '#00C853'},
    'Chest':  {'V1': '#B71C1C', 'V2': '#EF9A9A'},
}

def color_of(label):
    parts = label.split()           # e.g. ['Wrist', 'V1']
    return SITE_COLOR.get(parts[0], {}).get(parts[1], 'gray')

SITE_ORDER = ['Wrist V1', 'Wrist V2', 'Finger V1', 'Finger V2', 'Finger V3', 'Chest V1', 'Chest V2']

# Quality thresholds
THR_PI_GOOD  = 1.0    # PI% >= 1.0 → good
THR_PI_ACC   = 0.3    # PI% >= 0.3 → acceptable
THR_SNR_GOOD = 6.0    # SNR >= 6 dB → good


# ─── FILTERS ──────────────────────────────────────────────────────────────────

def _bp(x, fs, lo=0.5, hi=4.0, order=4):
    nyq = fs / 2.0
    b, a = sp.butter(order, [lo / nyq, min(hi / nyq, 0.9999)], btype='band')
    return sp.filtfilt(b, a, x)

def _hp(x, fs, co=0.05, order=2):
    nyq = fs / 2.0
    b, a = sp.butter(order, max(co / nyq, 1e-4), btype='high')
    return sp.filtfilt(b, a, x)

def _lp(x, fs, co=0.5, order=4):
    nyq = fs / 2.0
    b, a = sp.butter(order, min(co / nyq, 0.9999), btype='low')
    return sp.filtfilt(b, a, x)


# ─── DATA LOADER ──────────────────────────────────────────────────────────────

def load(label, relpath):
    """
    Returns dict with:
      t, ir, red, th3   – time and 3 PPG channel arrays (PPG rows only)
      t_acc, ax,ay,az   – accelerometer (Schema A) or empty arrays
      t_ev, sq,spo2,hr,pi,r_ratio  – sensor event rows (sparse)
      t_agc, agc1,agc2  – AGC LED current rows
      fs, schema, label, duration
    """
    path = os.path.join(BASE, relpath)
    if not os.path.isfile(path):
        print(f"  [MISSING] {path}"); return None

    print(f"  {label} ...", end=' ', flush=True)
    raw = pd.read_csv(path, low_memory=False)
    TS  = 'TIMESTAMP [s]'
    schema = 'A' if 'ACC_X' in raw.columns else 'B'

    # ── PPG rows ──────────────────────────────────────────────────────────────
    raw['_ir'] = pd.to_numeric(raw['PPG1_SUB1'], errors='coerce')
    ppg = raw[raw['_ir'].notna()].copy()
    ppg[TS]          = pd.to_numeric(ppg[TS], errors='coerce')
    ppg['PPG1_SUB2'] = pd.to_numeric(ppg['PPG1_SUB2'], errors='coerce').fillna(0.0)
    ppg['PPG1_SUB3'] = pd.to_numeric(ppg['PPG1_SUB3'], errors='coerce').fillna(0.0)
    ppg = ppg.dropna(subset=[TS]).sort_values(TS).reset_index(drop=True)

    fs = round(1.0 / ppg[TS].diff().dropna().median())
    t0 = ppg[TS].iloc[0]
    t   = (ppg[TS] - t0).values
    ir  = ppg['_ir'].values.astype(float)
    red = ppg['PPG1_SUB2'].values.astype(float)
    th3 = ppg['PPG1_SUB3'].values.astype(float)

    # zero = AGC settling; mask as NaN
    red = np.where(red == 0, np.nan, red)
    th3 = np.where(th3 == 0, np.nan, th3)

    # ── ACC rows ──────────────────────────────────────────────────────────────
    t_acc = ax = ay = az = np.array([])
    if schema == 'A':
        raw['_ax'] = pd.to_numeric(raw.get('ACC_X', pd.Series()), errors='coerce')
        acc = raw[raw['_ax'].notna()].copy()
        acc[TS] = pd.to_numeric(acc[TS], errors='coerce')
        acc = acc.dropna(subset=[TS]).sort_values(TS)
        t_acc = (acc[TS] - t0).values
        ax = pd.to_numeric(acc['ACC_X'], errors='coerce').values.astype(float)
        ay = pd.to_numeric(acc['ACC_Y'], errors='coerce').values.astype(float)
        az = pd.to_numeric(acc['ACC_Z'], errors='coerce').values.astype(float)

    # ── Sensor event rows (SIGNAL_QUALITY populated) ───────────────────────
    SQ = 'SPO2: SIGNAL_QUALITY'
    t_ev = sq = spo2_ev = hr_ev = pi_ev = r_ev = np.array([])
    if SQ in raw.columns:
        raw['_sq'] = pd.to_numeric(raw[SQ], errors='coerce')
        ev = raw[raw['_sq'].notna()].copy()
        ev[TS] = pd.to_numeric(ev[TS], errors='coerce')
        ev = ev.dropna(subset=[TS]).sort_values(TS)
        if len(ev):
            t_ev    = (ev[TS] - t0).values
            sq      = ev['_sq'].values.astype(float)
            spo2_ev = pd.to_numeric(ev.get('SPO2: SPO2 [%]',       np.nan), errors='coerce').values.astype(float)
            hr_ev   = pd.to_numeric(ev.get('SPO2: HEART_RATE [bpm]', np.nan), errors='coerce').values.astype(float)
            pi_ev   = pd.to_numeric(ev.get('SPO2: PI [%]',           np.nan), errors='coerce').values.astype(float)
            r_ev    = pd.to_numeric(ev.get('SPO2: R',                 np.nan), errors='coerce').values.astype(float)

    # ── AGC rows ──────────────────────────────────────────────────────────────
    t_agc = agc1 = agc2 = np.array([])
    if 'AGC1_LED_CURRENT' in raw.columns:
        raw['_a1'] = pd.to_numeric(raw['AGC1_LED_CURRENT'], errors='coerce')
        agc = raw[raw['_a1'].notna()].copy()
        agc[TS] = pd.to_numeric(agc[TS], errors='coerce')
        agc = agc.dropna(subset=[TS]).sort_values(TS)
        if len(agc):
            t_agc = (agc[TS] - t0).values
            agc1  = agc['_a1'].values.astype(float)
            agc2  = pd.to_numeric(agc.get('AGC2_LED_CURRENT', np.nan), errors='coerce').values.astype(float)

    dur = t[-1] if len(t) else 0
    print(f"fs={int(fs)}Hz  dur={dur:.0f}s  {len(t)} PPG rows  {len(t_ev)} sensor events")

    return dict(
        t=t, ir=ir, red=red, th3=th3,
        t_acc=t_acc, ax=ax, ay=ay, az=az,
        t_ev=t_ev, sq=sq, spo2_ev=spo2_ev, hr_ev=hr_ev, pi_ev=pi_ev, r_ev=r_ev,
        t_agc=t_agc, agc1=agc1, agc2=agc2,
        fs=int(fs), schema=schema, label=label,
        duration=dur, site=label.split()[0],
    )


# ─── SLIDING-WINDOW QUALITY METRICS ───────────────────────────────────────────

def sliding_pi(sig, t, fs, win_s=4.0, step_s=1.0):
    """
    Perfusion Index (%) in sliding windows.
    PI = (AC peak-to-peak / DC mean) * 100
    AC = bandpass filtered 0.5-4 Hz within window
    DC = mean of raw signal within window
    """
    n     = len(sig)
    win   = int(win_s  * fs)
    step  = int(step_s * fs)
    times, pis = [], []

    # Pre-filter whole signal once for speed
    try:
        sig_bp = _bp(np.nan_to_num(sig, nan=np.nanmean(sig[~np.isnan(sig)]) if np.any(~np.isnan(sig)) else 0), fs)
    except Exception:
        sig_bp = sig.copy()

    for start in range(0, n - win + 1, step):
        end    = start + win
        chunk  = sig[start:end]
        bp_c   = sig_bp[start:end]
        valid  = ~np.isnan(chunk)
        if np.sum(valid) < win // 2:
            continue
        dc = np.mean(chunk[valid])
        if dc <= 0:
            continue
        ac_pp = np.ptp(bp_c[valid])        # peak-to-peak of bandpass
        times.append(t[start + win // 2])
        pis.append(ac_pp / dc * 100.0)

    return np.array(times), np.array(pis)


def sliding_ac_amplitude(sig, t, fs, win_s=4.0, step_s=1.0):
    """Peak-to-peak AC amplitude (ADC counts) in sliding windows."""
    n    = len(sig)
    win  = int(win_s  * fs)
    step = int(step_s * fs)
    try:
        sig_bp = _bp(np.nan_to_num(sig, nan=np.nanmean(sig[~np.isnan(sig)]) if np.any(~np.isnan(sig)) else 0), fs)
    except Exception:
        sig_bp = sig.copy()

    times, amps = [], []
    for start in range(0, n - win + 1, step):
        end  = start + win
        bp_c = sig_bp[start:end]
        valid = ~np.isnan(sig[start:end])
        if np.sum(valid) < win // 2:
            continue
        times.append(t[start + win // 2])
        amps.append(np.ptp(bp_c[valid]))
    return np.array(times), np.array(amps)


def sliding_dc(sig, t, fs, win_s=10.0, step_s=2.0):
    """DC (mean) baseline in sliding windows — shows drift."""
    n    = len(sig)
    win  = int(win_s  * fs)
    step = int(step_s * fs)
    times, dcs = [], []
    for start in range(0, n - win + 1, step):
        end   = start + win
        chunk = sig[start:end]
        valid = ~np.isnan(chunk)
        if np.sum(valid) < win // 2:
            continue
        times.append(t[start + win // 2])
        dcs.append(np.mean(chunk[valid]))
    return np.array(times), np.array(dcs)


def sliding_snr(sig, t, fs, win_s=8.0, step_s=2.0):
    """
    SNR (dB) in sliding windows from Welch PSD.
    Signal band: 0.7-3.5 Hz (cardiac)
    Noise band:  4.0-8.0 Hz (above cardiac)
    """
    n    = len(sig)
    win  = int(win_s  * fs)
    step = int(step_s * fs)
    times, snrs = [], []
    for start in range(0, n - win + 1, step):
        end   = start + win
        chunk = np.nan_to_num(sig[start:end], nan=0.0)
        try:
            chunk_hp = _hp(chunk, fs, 0.05)
            f, p = sp.welch(chunk_hp, fs=fs, nperseg=min(len(chunk_hp), int(fs * 4)))
            s_pow = np.trapz(p[(f >= 0.7) & (f <= 3.5)], f[(f >= 0.7) & (f <= 3.5)])
            n_pow = np.trapz(p[(f >= 4.0) & (f <= min(8.0, fs / 2 - 0.1))],
                             f[(f >= 4.0) & (f <= min(8.0, fs / 2 - 0.1))])
            if n_pow > 0 and s_pow > 0:
                times.append(t[start + win // 2])
                snrs.append(10.0 * np.log10(s_pow / n_pow))
        except Exception:
            pass
    return np.array(times), np.array(snrs)


def beat_template(sig, t, fs, n_beats=20):
    """
    Extract N single heartbeat cycles and return mean template.
    Returns (template_x, template_mean, template_std, n_used)
    """
    try:
        sig_hp = _hp(np.nan_to_num(sig, nan=0.0), fs, 0.05)
        sig_bp = _bp(sig_hp, fs, 0.5, 4.0)
        dist   = int(fs * 0.4)
        thr    = 0.25 * np.nanmax(sig_bp)
        peaks, _ = sp.find_peaks(sig_bp, distance=dist, height=thr)
        if len(peaks) < 3:
            return None, None, None, 0
        # Use beats from middle of recording (skip first and last 10%)
        n_total = len(peaks)
        use = peaks[n_total // 10 : n_total - n_total // 10]
        if len(use) > n_beats:
            use = use[:n_beats]
        # Fixed window around each peak: 0.3s before, 0.5s after
        pre  = int(0.3 * fs)
        post = int(0.5 * fs)
        beats = []
        for pk in use:
            if pk - pre < 0 or pk + post >= len(sig_bp):
                continue
            beat = sig_hp[pk - pre: pk + post]
            beats.append(beat / np.ptp(beat) if np.ptp(beat) > 0 else beat)
        if len(beats) < 2:
            return None, None, None, 0
        L = min(len(b) for b in beats)
        arr = np.array([b[:L] for b in beats])
        x   = np.arange(L) / fs * 1000.0 - pre / fs * 1000.0   # ms from peak
        return x, np.mean(arr, axis=0), np.std(arr, axis=0), len(beats)
    except Exception:
        return None, None, None, 0


def _save(fig, fname):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {fname}")


# ─── PLOT 01: Perfusion Index over time ───────────────────────────────────────

def plot_01_pi(all_ds):
    print("\n[01] PI% over time ...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.suptitle(
        "Perfusion Index (PI%) Over Time — per Body Location\n"
        "PI = AC peak-to-peak / DC mean × 100  |  window=4s, step=1s\n"
        "Threshold lines: green=1.0% (good), orange=0.3% (acceptable)",
        fontsize=10, fontweight='bold'
    )
    site_axes = {'Wrist': axes[0], 'Finger': axes[1], 'Chest': axes[2]}

    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None: continue
        site = ds['site']
        ax   = site_axes[site]
        red  = ds['red']
        t    = ds['t']
        fs   = ds['fs']

        # Use RED channel for PI (strongest pulsatile signal)
        t_w, pi_w = sliding_pi(red, t, fs)
        if len(t_w) == 0: continue
        c = color_of(lbl)
        ax.plot(t_w, pi_w, color=c, lw=1.2, alpha=0.85, label=lbl)

        # Also plot sensor PI events if available
        if len(ds['t_ev']) > 0 and len(ds['pi_ev']) > 0:
            ok = np.isfinite(ds['pi_ev']) & (ds['pi_ev'] > 0)
            if np.any(ok):
                ax.scatter(ds['t_ev'][ok], ds['pi_ev'][ok], color=c,
                           marker='D', s=25, zorder=6, edgecolors='black',
                           linewidths=0.5, label=f'{lbl} (sensor PI)')

    for site, ax in site_axes.items():
        ax.axhline(THR_PI_GOOD, color='green',  lw=1.2, ls='--', alpha=0.7, label=f'{THR_PI_GOOD}% threshold (good)')
        ax.axhline(THR_PI_ACC,  color='orange', lw=1.2, ls='--', alpha=0.7, label=f'{THR_PI_ACC}% threshold (acceptable)')
        ax.set_title(site, fontsize=10, fontweight='bold',
                     color=list(SITE_COLOR[site].values())[0])
        ax.set_ylabel("Perfusion Index (%)", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds from recording start)", fontsize=9)
    fig.tight_layout()
    _save(fig, "01_pi_over_time.png")


# ─── PLOT 02: AC Amplitude over time ──────────────────────────────────────────

def plot_02_ac_amplitude(all_ds):
    print("\n[02] AC amplitude over time ...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(
        "AC Peak-to-Peak Amplitude Over Time — IR (left) vs RED (right)\n"
        "X = Time (seconds) | Y = Peak-to-Peak ADC counts in 4-second bandpass-filtered window (0.5-4 Hz)\n"
        "Larger amplitude = stronger pulsatile signal = better signal quality",
        fontsize=10, fontweight='bold'
    )
    sites = ['Wrist', 'Finger', 'Chest']
    row_axes = {s: (axes[i, 0], axes[i, 1]) for i, s in enumerate(sites)}

    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None: continue
        site = ds['site']
        ax_ir, ax_red = row_axes[site]
        t, ir, red, fs = ds['t'], ds['ir'], ds['red'], ds['fs']
        c = color_of(lbl)

        t_ir,  amp_ir  = sliding_ac_amplitude(ir,  t, fs)
        t_red, amp_red = sliding_ac_amplitude(red, t, fs)

        if len(t_ir)  > 0: ax_ir.plot( t_ir,  amp_ir,  color=c, lw=1.0, alpha=0.85, label=lbl)
        if len(t_red) > 0: ax_red.plot(t_red, amp_red, color=c, lw=1.0, alpha=0.85, label=lbl)

    for site, (ax_ir, ax_red) in row_axes.items():
        for ax, ch in [(ax_ir, 'IR ~850nm (SUB1)'), (ax_red, 'RED ~660nm (SUB2)')]:
            ax.set_title(f"{site} — {ch}", fontsize=9, fontweight='bold',
                         color=list(SITE_COLOR[site].values())[0])
            ax.set_ylabel("AC Peak-to-Peak (ADC counts)", fontsize=8)
            ax.legend(fontsize=7.5)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        ax_red.set_xlabel("Time (seconds)", fontsize=8)

    fig.tight_layout()
    _save(fig, "02_ac_amplitude_over_time.png")


# ─── PLOT 03: DC Baseline Drift ───────────────────────────────────────────────

def plot_03_dc_baseline(all_ds):
    print("\n[03] DC baseline drift ...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        "DC Baseline (Mean Signal Level) Over Time — IR and RED Channels\n"
        "X = Time (seconds) | Y = Mean ADC Counts in 10-second sliding window\n"
        "A stable flat line = good contact. Drift = motion or pressure change.",
        fontsize=10, fontweight='bold'
    )

    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None: continue
        t, ir, red, fs = ds['t'], ds['ir'], ds['red'], ds['fs']
        c = color_of(lbl)

        t_ir,  dc_ir  = sliding_dc(ir,  t, fs)
        t_red, dc_red = sliding_dc(red, t, fs)

        if len(t_ir)  > 0: axes[0].plot(t_ir,  dc_ir,  color=c, lw=1.0, alpha=0.8, label=lbl)
        if len(t_red) > 0: axes[1].plot(t_red, dc_red, color=c, lw=1.0, alpha=0.8, label=lbl)

    for ax, ch in zip(axes, ['IR ~850nm (SUB1)', 'RED ~660nm (SUB2)']):
        ax.set_title(ch, fontsize=10, fontweight='bold')
        ax.set_ylabel("Mean ADC Counts (DC level)", fontsize=9)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds from recording start)", fontsize=9)
    fig.tight_layout()
    _save(fig, "03_dc_baseline_over_time.png")


# ─── PLOT 04: SNR over time ────────────────────────────────────────────────────

def plot_04_snr(all_ds):
    print("\n[04] SNR over time ...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle(
        "Signal-to-Noise Ratio (SNR) Over Time — RED Channel\n"
        "X = Time (seconds) | Y = SNR (dB) in 8-second Welch PSD window\n"
        "SNR = cardiac band power (0.7-3.5 Hz) / noise power (4-8 Hz)\n"
        "Green dashed = 6 dB threshold (minimum for reliable HR detection)",
        fontsize=10, fontweight='bold'
    )
    site_axes = {'Wrist': axes[0], 'Finger': axes[1], 'Chest': axes[2]}

    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None: continue
        site = ds['site']
        ax   = site_axes[site]
        red, t, fs = ds['red'], ds['t'], ds['fs']
        c = color_of(lbl)

        t_s, snr_s = sliding_snr(red, t, fs)
        if len(t_s) > 0:
            ax.plot(t_s, snr_s, color=c, lw=1.2, alpha=0.85, label=lbl)
            # Fill above threshold green
            ax.fill_between(t_s, THR_SNR_GOOD, snr_s,
                            where=(snr_s >= THR_SNR_GOOD),
                            color=c, alpha=0.12)

    for site, ax in site_axes.items():
        ax.axhline(THR_SNR_GOOD, color='green', lw=1.3, ls='--', alpha=0.8,
                   label=f'{THR_SNR_GOOD} dB threshold')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_title(site, fontsize=10, fontweight='bold',
                     color=list(SITE_COLOR[site].values())[0])
        ax.set_ylabel("SNR (dB)", fontsize=8)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds from recording start)", fontsize=9)
    fig.tight_layout()
    _save(fig, "04_snr_over_time.png")


# ─── PLOT 05: Spectrogram ─────────────────────────────────────────────────────

def plot_05_spectrogram(all_ds):
    print("\n[05] Spectrograms ...")
    n = len(all_ds)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n))
    if n == 1: axes = [axes]
    fig.suptitle(
        "Time-Frequency Spectrogram of RED Channel (AC, highpass 0.05 Hz)\n"
        "X = Time (seconds) | Y = Frequency (Hz) | Color = Power (dB)\n"
        "Bright horizontal stripe at ~1 Hz = heart rate. Width = HR variability.",
        fontsize=10, fontweight='bold'
    )

    for ax, lbl in zip(axes, SITE_ORDER):
        ds = all_ds.get(lbl)
        if ds is None:
            ax.set_visible(False)
            continue
        red, t, fs = ds['red'], ds['t'], ds['fs']
        c = color_of(lbl)

        # Fill NaN for spectrogram
        red_f = np.where(np.isnan(red), 0.0, red)
        try:
            red_hp = _hp(red_f, fs, 0.05)
        except Exception:
            red_hp = red_f

        # Welch spectrogram
        nperseg  = min(len(red_hp), int(fs * 8))
        noverlap = int(nperseg * 0.75)
        f, t_spec, Sxx = sp.spectrogram(red_hp, fs=fs,
                                          nperseg=nperseg, noverlap=noverlap,
                                          window='hann', scaling='density')

        # Limit to 0-5 Hz and dB scale
        f_mask = f <= 5.0
        Sxx_db = 10 * np.log10(Sxx[f_mask] + 1e-12)

        im = ax.pcolormesh(t_spec + t[0], f[f_mask], Sxx_db,
                           shading='gouraud', cmap='inferno',
                           vmin=np.percentile(Sxx_db, 10),
                           vmax=np.percentile(Sxx_db, 99))
        plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.8)

        ax.set_ylabel("Frequency (Hz)", fontsize=8)
        ax.set_ylim(0, 5)
        ax.axhline(0.7, color='cyan',  lw=0.8, ls='--', alpha=0.7, label='HR band start 0.7 Hz')
        ax.axhline(3.5, color='cyan',  lw=0.8, ls='--', alpha=0.7)
        ax.axhline(0.1, color='white', lw=0.8, ls=':', alpha=0.5, label='RR band 0.1-0.5 Hz')
        ax.axhline(0.5, color='white', lw=0.8, ls=':', alpha=0.5)
        ax.set_title(f"{lbl}  |  {ds['site']}  |  Fs={fs} Hz  |  Duration={ds['duration']:.0f}s",
                     fontsize=9, fontweight='bold')
        ax.legend(fontsize=6.5, loc='upper right')
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (seconds from recording start)", fontsize=9)
    fig.tight_layout()
    _save(fig, "05_spectrogram.png")


# ─── PLOT 06: PSD per site ────────────────────────────────────────────────────

def plot_06_psd(all_ds):
    print("\n[06] PSD per site ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Power Spectral Density (Welch) of RED Channel — by Body Location\n"
        "X = Frequency (Hz) | Y = PSD (ADC counts squared / Hz, log scale)\n"
        "A sharp tall peak in the HR band = clean, detectable heartbeat signal.",
        fontsize=10, fontweight='bold'
    )
    site_axes = {'Wrist': axes[0], 'Finger': axes[1], 'Chest': axes[2]}

    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None: continue
        site = ds['site']
        ax   = site_axes[site]
        red, t, fs = ds['red'], ds['t'], ds['fs']
        c = color_of(lbl)

        red_f  = np.where(np.isnan(red), 0.0, red)
        red_hp = _hp(red_f, fs, 0.05)
        nperseg = min(len(red_hp), int(fs * 16))
        f, p   = sp.welch(red_hp, fs=fs, nperseg=nperseg)

        mask_hr  = (f >= 0.7) & (f <= 3.5)
        dom_f    = f[mask_hr][np.argmax(p[mask_hr])] if np.any(mask_hr) else np.nan
        dom_bpm  = dom_f * 60 if np.isfinite(dom_f) else 0

        ax.semilogy(f, p, color=c, lw=1.4, alpha=0.85,
                    label=f"{lbl}  HR~{dom_bpm:.0f} BPM")
        if np.isfinite(dom_f):
            ax.axvline(dom_f, color=c, lw=1.0, ls='--', alpha=0.6)

    for site, ax in site_axes.items():
        ax.axvspan(0.1, 0.5, alpha=0.07, color='purple')
        ax.axvspan(0.7, 3.5, alpha=0.07, color='green')
        ax.text(0.3,  ax.get_ylim()[0] * 2, 'RR\nband', fontsize=6.5, color='purple', ha='center')
        ax.text(2.1,  ax.get_ylim()[0] * 2, 'HR band', fontsize=6.5, color='green',  ha='center')
        ax.set_xlim(0, 5)
        ax.set_title(site, fontsize=10, fontweight='bold',
                     color=list(SITE_COLOR[site].values())[0])
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD (ADC² / Hz)", fontsize=8)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "06_psd_per_dataset.png")


# ─── PLOT 07: Beat Template ────────────────────────────────────────────────────

def plot_07_beat_template(all_ds):
    print("\n[07] Beat templates ...")
    n = len(SITE_ORDER)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.suptitle(
        "Averaged Heartbeat Template — IR Channel (AC-filtered, normalised to [-1, 1])\n"
        "X = Time (ms relative to systolic peak) | Y = Normalised amplitude (a.u.)\n"
        "Sharp clear peak + dicrotic notch = high-quality PPG waveform.",
        fontsize=10, fontweight='bold'
    )
    axes = axes.flat

    for ax, lbl in zip(axes, SITE_ORDER):
        ds = all_ds.get(lbl)
        site = ds['site'] if ds else 'N/A'
        c    = color_of(lbl)

        if ds is None:
            ax.set_visible(False)
            continue

        x, tmean, tstd, n_used = beat_template(ds['ir'], ds['t'], ds['fs'])

        if x is None or n_used < 2:
            ax.text(0.5, 0.5, f'{lbl}\nInsufficient beats detected',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
            ax.set_title(lbl, fontsize=9)
            continue

        ax.plot(x, tmean, color=c, lw=1.8, label=f'Mean ({n_used} beats)')
        ax.fill_between(x, tmean - tstd, tmean + tstd, color=c, alpha=0.2, label='±1 SD')
        ax.axvline(0, color='orange', lw=1.0, ls='--', alpha=0.7, label='Systolic peak')
        ax.axhline(0, color='gray',   lw=0.6, ls=':')

        consistency = 1.0 - np.mean(tstd[np.abs(x) < 200]) if len(tstd) else np.nan
        ax.set_title(f"{lbl}  |  Consistency={consistency:.2f}", fontsize=8.5)
        ax.set_xlabel("Time from peak (ms)", fontsize=7.5)
        ax.set_ylabel("Norm. amplitude", fontsize=7.5)
        ax.legend(fontsize=6.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in list(axes)[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, "07_beat_template.png")


# ─── PLOT 08: IR vs RED Correlation ───────────────────────────────────────────

def plot_08_ir_red_corr(all_ds):
    print("\n[08] IR vs RED correlation ...")
    n = len(SITE_ORDER)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.suptitle(
        "IR vs RED AC Signal Cross-Plot (first 60 seconds)\n"
        "X = IR AC (ADC counts) | Y = RED AC (ADC counts)\n"
        "A tight ellipse / line = IR and RED pulses are correlated (good quality).\n"
        "Scattered cloud = noise dominant / poor contact.",
        fontsize=10, fontweight='bold'
    )
    axes = axes.flat

    for ax, lbl in zip(axes, SITE_ORDER):
        ds = all_ds.get(lbl)
        if ds is None:
            ax.set_visible(False)
            continue
        t, ir, red, fs = ds['t'], ds['ir'], ds['red'], ds['fs']
        c = color_of(lbl)

        # First 60s, skip first 2s (AGC settling)
        mask = (t >= 2.0) & (t <= 62.0)
        ir_m  = ir[mask]
        red_m = red[mask]
        ok    = ~np.isnan(ir_m) & ~np.isnan(red_m)

        if np.sum(ok) < 50:
            ax.text(0.5, 0.5, 'Insufficient valid data\n(RED=0 during AGC settling)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(lbl, fontsize=9)
            continue

        ir_ac  = _hp(np.nan_to_num(ir_m[ok],  nan=0.0), fs, 0.05)
        red_ac = _hp(np.nan_to_num(red_m[ok], nan=0.0), fs, 0.05)

        try:
            r, _ = __import__('scipy').stats.pearsonr(ir_ac, red_ac)
        except Exception:
            r = np.nan

        ax.scatter(ir_ac[::5], red_ac[::5], color=c, s=2, alpha=0.4)
        ax.set_title(f"{lbl}  |  Pearson r = {r:.3f}", fontsize=8.5)
        ax.set_xlabel("IR AC (ADC counts)", fontsize=7.5)
        ax.set_ylabel("RED AC (ADC counts)", fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # Quality note
        badge = 'GOOD' if abs(r) >= 0.7 else ('FAIR' if abs(r) >= 0.4 else 'POOR')
        badge_col = 'green' if badge == 'GOOD' else ('orange' if badge == 'FAIR' else 'red')
        ax.text(0.97, 0.97, badge, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_col, alpha=0.85))

    for ax in list(axes)[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, "08_ir_red_correlation.png")


# ─── PLOT 09: Sensor Events (PI, R, SpO2, HR from firmware) ───────────────────

def plot_09_sensor_events(all_ds):
    print("\n[09] Sensor event data ...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    fig.suptitle(
        "Sensor-Reported Quality Metrics (Sparse Event Rows)\n"
        "These are the AS7058 firmware outputs — only appear when firmware computes a value.\n"
        "Empty regions = firmware was not outputting (raw streaming mode).",
        fontsize=10, fontweight='bold'
    )
    cols_axes = [
        ('pi_ev',   axes[0], 'Perfusion Index PI (%)',           [0, 5]),
        ('r_ev',    axes[1], 'R-Ratio  (for SpO2 calculation)',  [0, 3]),
        ('spo2_ev', axes[2], 'SpO2 (%)',                         [85, 102]),
        ('hr_ev',   axes[3], 'Heart Rate (BPM)',                 [40, 180]),
    ]

    any_data = False
    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None or len(ds['t_ev']) == 0: continue
        c = color_of(lbl)
        t_ev = ds['t_ev']

        for key, ax, ylabel, ylim in cols_axes:
            vals = ds[key]
            if len(vals) == 0: continue
            ok = np.isfinite(vals) & (vals > 0)
            if not np.any(ok): continue
            any_data = True
            ax.scatter(t_ev[ok], vals[ok], color=c, s=20, alpha=0.7,
                       label=lbl, edgecolors='black', linewidths=0.3)

    for key, ax, ylabel, ylim in cols_axes:
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, ncol=3)
        if key == 'spo2_ev':
            ax.axhline(95, color='red', lw=1.0, ls=':', label='95% clinical threshold')

    if not any_data:
        for ax in axes:
            ax.text(0.5, 0.5,
                    'No sensor event data found\n'
                    '(AS7058 firmware in raw streaming mode — SpO2 algorithm not running)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='gray', style='italic')

    axes[-1].set_xlabel("Time (seconds from recording start)", fontsize=9)
    fig.tight_layout()
    _save(fig, "09_sensor_events.png")


# ─── PLOT 10: AGC Settling ────────────────────────────────────────────────────

def plot_10_agc(all_ds):
    print("\n[10] AGC settling ...")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    fig.suptitle(
        "AGC LED Drive Current Over Time — Sensor Auto-Calibration\n"
        "X = Time (seconds) | Y = LED Current (ADC LSB, 1 LSB ~ 0.4 mA for AS7058)\n"
        "Lower settled current = sensor found strong reflected light = better tissue-optical coupling.\n"
        "High settled current = sensor struggling to get enough signal back = poor contact.",
        fontsize=10, fontweight='bold'
    )
    site_axes = {'Wrist': axes[0], 'Finger': axes[1], 'Chest': axes[2]}

    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None or len(ds['t_agc']) == 0: continue
        site = ds['site']
        ax   = site_axes[site]
        c    = color_of(lbl)
        t_agc, agc1, agc2 = ds['t_agc'], ds['agc1'], ds['agc2']

        ls = '-' if 'V1' in lbl else '--'
        ax.step(t_agc, agc1, color=c, lw=1.5, ls=ls, where='post', label=f'{lbl} AGC1 (IR LED)')
        ok2 = np.isfinite(agc2) & (agc2 > 0)
        if np.any(ok2):
            ax.step(t_agc[ok2], agc2[ok2], color=c, lw=0.9, ls=ls,
                    alpha=0.5, where='post', label=f'{lbl} AGC2 (RED LED)')

        # Annotate final settled value
        if len(agc1) > 0:
            final = agc1[-1]
            ax.annotate(f'settled={final:.0f}',
                        xy=(t_agc[-1], final), xytext=(-30, 8),
                        textcoords='offset points', fontsize=7, color=c,
                        arrowprops=dict(arrowstyle='->', color=c, lw=0.8))

    for site, ax in site_axes.items():
        ax.set_title(site, fontsize=10, fontweight='bold',
                     color=list(SITE_COLOR[site].values())[0])
        ax.set_ylabel("LED Current (LSB)", fontsize=8)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if len(ax.lines) == 0:
            ax.text(0.5, 0.5, 'No AGC data (Schema B file)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')

    axes[-1].set_xlabel("Time (seconds from recording start)", fontsize=9)
    fig.tight_layout()
    _save(fig, "10_agc_settling.png")


# ─── PLOT 11: Quality Heatmap ─────────────────────────────────────────────────

def build_summary_metrics(all_ds):
    """Compute one scalar per dataset per metric for the heatmap."""
    rows = []
    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None: continue
        t, ir, red, fs = ds['t'], ds['ir'], ds['red'], ds['fs']

        # Use data after AGC settling
        ok = t >= 2.0
        red_ok = red[ok]; red_ok = red_ok[~np.isnan(red_ok)]
        ir_ok  = ir[ok]

        # PI from RED
        _, pi_arr = sliding_pi(red, t, fs)
        med_pi = float(np.median(pi_arr)) if len(pi_arr) else np.nan

        # AC amplitude RED
        _, amp_arr = sliding_ac_amplitude(red, t, fs)
        med_amp_red = float(np.median(amp_arr)) if len(amp_arr) else np.nan

        # AC amplitude IR
        _, amp_ir_arr = sliding_ac_amplitude(ir, t, fs)
        med_amp_ir = float(np.median(amp_ir_arr)) if len(amp_ir_arr) else np.nan

        # SNR RED
        _, snr_arr = sliding_snr(red, t, fs)
        med_snr = float(np.median(snr_arr)) if len(snr_arr) else np.nan

        # DC RED
        med_dc_red = float(np.mean(red_ok)) if len(red_ok) else np.nan

        # AGC final
        agc_final = float(ds['agc1'][-1]) if len(ds['agc1']) > 0 else np.nan

        # Sensor PI (if any)
        sensor_pi = float(np.nanmedian(ds['pi_ev'])) if (len(ds['pi_ev']) > 0 and np.any(np.isfinite(ds['pi_ev']))) else np.nan

        # IR-RED correlation
        try:
            mask60 = (t >= 2.0) & (t <= 62.0)
            ir_m, red_m = ir[mask60], red[mask60]
            ok_m = ~np.isnan(ir_m) & ~np.isnan(red_m)
            if np.sum(ok_m) > 50:
                ir_ac  = _hp(np.nan_to_num(ir_m[ok_m],  nan=0.0), fs, 0.05)
                red_ac = _hp(np.nan_to_num(red_m[ok_m], nan=0.0), fs, 0.05)
                r_corr, _ = __import__('scipy').stats.pearsonr(ir_ac, red_ac)
            else:
                r_corr = np.nan
        except Exception:
            r_corr = np.nan

        rows.append({
            'Dataset':          lbl,
            'Site':             ds['site'],
            'PI% (computed)':   round(med_pi, 3),
            'PI% (sensor)':     round(sensor_pi, 3) if np.isfinite(sensor_pi) else np.nan,
            'AC amp RED (ADC)': round(med_amp_red, 0),
            'AC amp IR (ADC)':  round(med_amp_ir, 0),
            'SNR (dB)':         round(med_snr, 1),
            'DC RED (ADC)':     round(med_dc_red, 0),
            'AGC1 final (LSB)': round(agc_final, 0) if np.isfinite(agc_final) else np.nan,
            'IR-RED corr (r)':  round(r_corr, 3) if np.isfinite(r_corr) else np.nan,
            'Duration (s)':     round(ds['duration'], 0),
            'Fs (Hz)':          ds['fs'],
        })

    return pd.DataFrame(rows)


def plot_11_heatmap(summary_df):
    print("\n[11] Quality heatmap ...")

    metric_cols = [
        'PI% (computed)', 'PI% (sensor)', 'AC amp RED (ADC)',
        'AC amp IR (ADC)', 'SNR (dB)', 'IR-RED corr (r)',
        'AGC1 final (LSB)', 'Duration (s)',
    ]
    # Higher-is-better cols (normalise 0→1, 1=best)
    higher_better = {'PI% (computed)', 'PI% (sensor)', 'AC amp RED (ADC)',
                     'AC amp IR (ADC)', 'SNR (dB)', 'IR-RED corr (r)', 'Duration (s)'}
    # Lower-is-better: AGC final

    data = summary_df.set_index('Dataset')[metric_cols].copy()

    # Normalise each column to 0-1
    norm = data.copy()
    for col in metric_cols:
        col_vals = data[col].values.astype(float)
        mn, mx = np.nanmin(col_vals), np.nanmax(col_vals)
        if mx == mn:
            norm[col] = 0.5
        else:
            if col in higher_better:
                norm[col] = (col_vals - mn) / (mx - mn)
            else:
                norm[col] = 1.0 - (col_vals - mn) / (mx - mn)

    norm_arr = norm.values.astype(float)

    fig, (ax_heat, ax_table) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={'width_ratios': [2, 1]}
    )
    fig.suptitle(
        "Signal Quality Heatmap — All Datasets\n"
        "Left: normalised 0-1 per metric (green=best, red=worst)\n"
        "Right: actual values",
        fontsize=11, fontweight='bold'
    )

    im = ax_heat.imshow(norm_arr, aspect='auto', cmap='RdYlGn',
                        vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax_heat, label='Normalised Quality (0=worst, 1=best)', shrink=0.8)

    ax_heat.set_xticks(range(len(metric_cols)))
    ax_heat.set_xticklabels(metric_cols, rotation=35, ha='right', fontsize=8)
    ax_heat.set_yticks(range(len(norm.index)))
    ax_heat.set_yticklabels(norm.index, fontsize=9)

    # Value annotations on heatmap
    for i in range(norm_arr.shape[0]):
        for j in range(norm_arr.shape[1]):
            v = data.values[i, j]
            txt = f'{v:.1f}' if np.isfinite(v) else 'N/A'
            ax_heat.text(j, i, txt, ha='center', va='center', fontsize=7,
                         color='black' if 0.2 < norm_arr[i, j] < 0.8 else 'white')

    # Site color bands on y-axis
    for i, lbl in enumerate(norm.index):
        c = color_of(lbl)
        ax_heat.add_patch(plt.Rectangle((-0.5, i - 0.5), 0.25, 1.0,
                                         color=c, clip_on=False))

    # Table (right)
    table_data = summary_df[['Dataset', 'Site', 'PI% (computed)', 'SNR (dB)',
                              'AC amp RED (ADC)', 'AGC1 final (LSB)', 'Duration (s)']].copy()
    table_data = table_data.set_index('Dataset')

    ax_table.axis('off')
    tbl = ax_table.table(
        cellText=table_data.values,
        colLabels=table_data.columns.tolist(),
        rowLabels=table_data.index.tolist(),
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.5)

    # Colour rows by PI quality
    for i, lbl in enumerate(table_data.index):
        pi_val = table_data.loc[lbl, 'PI% (computed)']
        if np.isfinite(pi_val):
            bg = '#C8E6C9' if pi_val >= THR_PI_GOOD else ('#FFF9C4' if pi_val >= THR_PI_ACC else '#FFCDD2')
            for j in range(len(table_data.columns)):
                tbl[i + 1, j].set_facecolor(bg)

    fig.tight_layout()
    _save(fig, "11_quality_heatmap.png")
    return summary_df


# ─── PLOT 12: AC Waveform Overlay (30s) ──────────────────────────────────────

def plot_12_waveform_overlay(all_ds):
    print("\n[12] Waveform overlay ...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        "AC-Filtered RED Waveform — First 30 Seconds After AGC Settling\n"
        "X = Time (seconds) | Y = DELTA ADC Counts (baseline removed, highpass 0.05 Hz)\n"
        "Each peak = one heartbeat. Amplitude directly shows signal strength per site.",
        fontsize=10, fontweight='bold'
    )
    site_axes = {'Wrist': axes[0], 'Finger': axes[1], 'Chest': axes[2]}

    for lbl in SITE_ORDER:
        ds = all_ds.get(lbl)
        if ds is None: continue
        site = ds['site']
        ax   = site_axes[site]
        t, red, fs = ds['t'], ds['red'], ds['fs']
        c = color_of(lbl)

        # 30s window after AGC settling (start at 2s)
        mask = (t >= 2.0) & (t <= 32.0)
        t_w   = t[mask]
        red_w = red[mask]
        ok    = ~np.isnan(red_w)

        if np.sum(ok) < int(fs * 2):
            ax.text(0.5, 0.5, f'{lbl}: no RED data in window',
                    ha='center', va='center', transform=ax.transAxes, color='gray')
            continue

        fill_red = red_w.copy()
        fill_red[~ok] = np.nanmean(red_w[ok])
        try:
            red_ac = _hp(fill_red, fs, 0.05)
        except Exception:
            red_ac = fill_red - np.nanmean(fill_red)

        red_ac[~ok] = np.nan
        t_disp = t_w - t_w[0]

        ax.plot(t_disp, red_ac, color=c, lw=0.7, alpha=0.85, label=lbl)
        ax.axhline(0, color='gray', lw=0.5, ls='--')

    for site, ax in site_axes.items():
        ax.set_title(site, fontsize=10, fontweight='bold',
                     color=list(SITE_COLOR[site].values())[0])
        ax.set_ylabel("Delta ADC Counts\n(AC component)", fontsize=8)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds, starting 2s after recording start)", fontsize=9)
    fig.tight_layout()
    _save(fig, "12_waveform_overlay_30s.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 60)
    print(' PPG Signal Quality Analysis')
    print('=' * 60)

    print('\n[1] Loading all datasets ...')
    all_ds = {}
    for lbl, path in DATASETS.items():
        ds = load(lbl, path)
        if ds: all_ds[lbl] = ds

    print('\n[2] Generating quality plots ...')
    plot_01_pi(all_ds)
    plot_02_ac_amplitude(all_ds)
    plot_03_dc_baseline(all_ds)
    plot_04_snr(all_ds)
    plot_05_spectrogram(all_ds)
    plot_06_psd(all_ds)
    plot_07_beat_template(all_ds)
    plot_08_ir_red_corr(all_ds)
    plot_09_sensor_events(all_ds)
    plot_10_agc(all_ds)

    print('\n[3] Computing summary metrics ...')
    summary = build_summary_metrics(all_ds)
    plot_11_heatmap(summary)
    plot_12_waveform_overlay(all_ds)

    # Console summary table
    print('\n' + '=' * 85)
    print(f"{'Dataset':<12} {'PI%(comp)':<11} {'PI%(sens)':<11} {'SNR dB':<9} "
          f"{'AC RED':<10} {'AC IR':<9} {'AGC1':<8} {'r(IR,RED)':<11} {'Dur(s)'}")
    print('-' * 85)
    for _, row in summary.iterrows():
        def f(v, d=2): return f'{v:.{d}f}' if pd.notna(v) else 'N/A'
        print(f"{row['Dataset']:<12} {f(row['PI% (computed)']):<11} {f(row['PI% (sensor)']):<11} "
              f"{f(row['SNR (dB)'], 1):<9} {f(row['AC amp RED (ADC)'], 0):<10} "
              f"{f(row['AC amp IR (ADC)'], 0):<9} {f(row['AGC1 final (LSB)'], 0):<8} "
              f"{f(row['IR-RED corr (r)'], 3):<11} {f(row['Duration (s)'], 0)}")
    print('=' * 85)

    print(f'\n All plots saved to: {OUT}')
    print(' Done.\n')


if __name__ == '__main__':
    main()
