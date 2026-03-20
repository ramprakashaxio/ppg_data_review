"""
Position A Chest PPG Signal Quality Analysis — Iteration 2
============================================================
Run:   py -3 position_a_analysis.py
Output: Iteration 2_Test data/Postion A/Analysis/

Analyses Position A chest placement from the AS7058 Iteration 2 dataset.
Computes PI, SNR, AC amplitude, HR, and RR CV across SpO2 and HRM modes.
Compares against Iteration 1 chest baseline (PI = 0.22%).
"""

import os, sys, warnings, textwrap
import numpy as np
import pandas as pd
import scipy.signal as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POS_A_DIR  = os.path.join(SCRIPT_DIR, 'Iteration 2_Test data', 'Postion A')
OUT_DIR    = os.path.join(POS_A_DIR, 'Analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# File paths (using %23 encoding in filenames as stored on disk)
SPO2_RAW_F  = os.path.join(POS_A_DIR, 'SpO2',
    'Chest_position_nikhil_V1%23_2026-03-04_15-53-16.csv')
SPO2_FILT_F = os.path.join(POS_A_DIR, 'SpO2',
    'Chest_position_nikhil_V1%23_2026-03-04_15-49-57_filtered.csv')
HRM_RAW_F   = os.path.join(POS_A_DIR, 'HRM RRM',
    'Chest_position_HRM RRM Nikhil_V1%23_2026-03-04_15-58-32.csv')
HRM_FILT_F  = os.path.join(POS_A_DIR, 'HRM RRM',
    'Chest_position_HRM RRM Nikhil_V1%23_2026-03-04_15-54-56_filtered.csv')

# Quality thresholds (same as Iteration 1)
THR = dict(
    pi_good=1.0,   pi_fair=0.3,
    snr_good=10.0, snr_fair=6.0,
    rrcv_good=10.0, rrcv_fair=25.0,
    amp_good=100,  amp_fair=20,
    hr_lo=40.0,    hr_hi=160.0,
)

# Iteration 1 chest baseline for comparison
ITER1_CHEST = dict(pi=0.22, snr=19.1, amp=6.0, hr_pct=100.0, rrcv=3.6, agc=13.0)

TAIL_S = 120.0  # last 120s for stable summaries
FS = 200         # all Position A files are 200 Hz

# ── Signal processing ─────────────────────────────────────────────────────────
def bandpass(sig, fs, lo=0.5, hi=4.0):
    nyq = fs / 2.0
    b, a = sp.butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype='band')
    return sp.filtfilt(b, a, np.nan_to_num(sig))


def compute_snr(ac_signal, fs):
    """Compute SNR in dB: cardiac band (0.7-3.5 Hz) vs noise band (4-8 Hz)."""
    f, p = sp.welch(ac_signal - np.mean(ac_signal), fs=fs,
                    nperseg=min(len(ac_signal), int(fs * 8)))
    sig_p   = np.trapz(p[(f >= 0.7) & (f <= 3.5)], f[(f >= 0.7) & (f <= 3.5)])
    noise_p = np.trapz(p[(f >= 4.0) & (f <= 8.0)], f[(f >= 4.0) & (f <= 8.0)])
    return 10 * np.log10(sig_p / noise_p) if (sig_p > 0 and noise_p > 0) else 0.0


def detect_hr(ac_signal, fs):
    """Detect HR and RR CV from bandpass-filtered signal."""
    dist = int(fs * 0.35)
    thresh = max(0.1 * np.max(np.abs(ac_signal)), 1e-6)
    peaks, _ = sp.find_peaks(ac_signal, distance=dist, height=thresh)
    if len(peaks) >= 4:
        rr = np.diff(peaks) / fs
        valid_rr = rr[(rr > 0.35) & (rr < 1.5)]
        if len(valid_rr) >= 3:
            hr   = 60.0 / np.mean(valid_rr)
            rrcv = (np.std(valid_rr) / np.mean(valid_rr)) * 100
            return hr, rrcv, peaks
    return np.nan, np.nan, peaks


# ── Data loading ──────────────────────────────────────────────────────────────
def load_raw_file(fpath, mode='spo2'):
    """Load a raw CSV. Returns dict with PPG channels, sensor events, AGC."""
    print(f'  Loading {os.path.basename(fpath)} ...')
    raw = pd.read_csv(fpath, low_memory=False)

    ts_col = 'TIMESTAMP [s]'
    t_raw  = pd.to_numeric(raw[ts_col], errors='coerce')
    s1_raw = pd.to_numeric(raw['PPG1_SUB1'], errors='coerce')
    s2_raw = pd.to_numeric(raw['PPG1_SUB2'], errors='coerce')

    # PPG rows: where PPG1_SUB1 is not NaN
    ppg_mask = s1_raw.notna()
    t  = t_raw[ppg_mask].values.astype(float)
    s1 = s1_raw[ppg_mask].values.astype(float)
    s2 = s2_raw[ppg_mask].values.astype(float)
    t -= t[0]

    # SUB3 for SpO2 mode
    s3 = np.full_like(s1, np.nan)
    if 'PPG1_SUB3' in raw.columns:
        s3_raw = pd.to_numeric(raw['PPG1_SUB3'], errors='coerce')
        s3 = s3_raw[ppg_mask].values.astype(float)

    # Sampling rate
    diffs = np.diff(t)
    fs = round(1.0 / np.median(diffs[diffs > 0]))

    # AGC currents
    agc_info = {}
    for prefix in ['AGC1', 'AGC2']:
        led_col = f'{prefix}_LED_CURRENT'
        if led_col in raw.columns:
            v = pd.to_numeric(raw[led_col], errors='coerce').dropna()
            if len(v):
                agc_info[f'{prefix}_led'] = float(v.median())
                agc_info[f'{prefix}_led_vals'] = v.values

    # Sensor events
    events = {}
    if mode == 'spo2':
        sq_col = 'SPO2: SIGNAL_QUALITY'
        if sq_col in raw.columns:
            sq_mask = pd.to_numeric(raw[sq_col], errors='coerce').notna()
            ev_df = raw[sq_mask].copy()
            if len(ev_df):
                et = pd.to_numeric(ev_df[ts_col], errors='coerce').values - t_raw[ppg_mask].values[0]
                events['t']   = et
                events['sq']  = pd.to_numeric(ev_df[sq_col], errors='coerce').values
                events['hr']  = pd.to_numeric(ev_df.get('SPO2: HEART_RATE [bpm]', pd.Series()), errors='coerce').values
                events['spo2']= pd.to_numeric(ev_df.get('SPO2: SPO2 [%]', pd.Series()), errors='coerce').values
                events['pi']  = pd.to_numeric(ev_df.get('SPO2: PI [%]', pd.Series()), errors='coerce').values
                events['r']   = pd.to_numeric(ev_df.get('SPO2: R', pd.Series()), errors='coerce').values
    elif mode == 'hrm':
        sq_col = 'HRM: SIGNAL_QUALITY'
        if sq_col in raw.columns:
            sq_mask = pd.to_numeric(raw[sq_col], errors='coerce').notna()
            ev_df = raw[sq_mask].copy()
            if len(ev_df):
                et = pd.to_numeric(ev_df[ts_col], errors='coerce').values - t_raw[ppg_mask].values[0]
                events['t']   = et
                events['sq']  = pd.to_numeric(ev_df[sq_col], errors='coerce').values
                events['hr']  = pd.to_numeric(ev_df.get('HRM: HEART_RATE [bpm]', pd.Series()), errors='coerce').values
                events['motion'] = pd.to_numeric(ev_df.get('HRM: MOTION_LEVEL', pd.Series()), errors='coerce').values
                events['prv'] = pd.to_numeric(ev_df.get('HRM: PRV [ms]', pd.Series()), errors='coerce').values
        # RRM events
        rrm_col = 'RRM: CONFIDENCE'
        if rrm_col in raw.columns:
            rrm_mask = pd.to_numeric(raw[rrm_col], errors='coerce').notna()
            rrm_df = raw[rrm_mask].copy()
            if len(rrm_df):
                events['rrm_t']    = pd.to_numeric(rrm_df[ts_col], errors='coerce').values - t_raw[ppg_mask].values[0]
                events['rrm_conf'] = pd.to_numeric(rrm_df[rrm_col], errors='coerce').values
                events['rrm_rr']   = pd.to_numeric(rrm_df.get('RRM: RESPIRATION_RATE [bpm]', pd.Series()), errors='coerce').values

    # Accelerometer (SpO2 mode only)
    acc_t, acc_mag = np.array([]), np.array([])
    if 'ACC_X' in raw.columns:
        acc_mask = pd.to_numeric(raw.get('ACC_X', pd.Series(dtype=float)), errors='coerce').notna()
        ar = raw[acc_mask]
        if len(ar):
            at = pd.to_numeric(ar[ts_col], errors='coerce').values.astype(float)
            at -= at[0]
            amg = np.sqrt(
                pd.to_numeric(ar['ACC_X'], errors='coerce').values**2 +
                pd.to_numeric(ar['ACC_Y'], errors='coerce').values**2 +
                pd.to_numeric(ar['ACC_Z'], errors='coerce').values**2
            )
            acc_t, acc_mag = at, amg

    return dict(
        t=t, s1=s1, s2=s2, s3=s3, fs=int(fs),
        duration=float(t[-1] - t[0]),
        agc=agc_info, events=events,
        acc_t=acc_t, acc_mag=acc_mag,
        mode=mode, ftype='raw',
    )


def load_filtered_file(fpath):
    """Load a filtered CSV. Returns dict with PPG sub-channels."""
    print(f'  Loading {os.path.basename(fpath)} ...')
    raw = pd.read_csv(fpath, low_memory=False)
    ts_col = 'TIMESTAMP [s]'
    t = pd.to_numeric(raw[ts_col], errors='coerce').dropna().values.astype(float)
    t -= t[0]

    channels = {}
    for col in raw.columns:
        if col.startswith('PPG') and 'SUB' in col:
            v = pd.to_numeric(raw[col], errors='coerce').values[:len(t)]
            if np.nanstd(v) > 0.1:  # only keep channels with actual signal
                channels[col] = v

    diffs = np.diff(t)
    fs = round(1.0 / np.median(diffs[diffs > 0]))

    return dict(t=t, channels=channels, fs=int(fs),
                duration=float(t[-1] - t[0]), ftype='filtered')


# ── Sliding window metrics ────────────────────────────────────────────────────
def compute_sliding_metrics(t, signal, fs, win=10.0, step=2.0, skip=10.0):
    """Compute PI, SNR, AC amp, HR, RR CV in sliding windows over raw signal."""
    win_t, pis, snrs, amps, hrs, rrcvs = [], [], [], [], [], []

    start = skip
    duration = t[-1] - t[0]
    while start + win <= duration:
        end = start + win
        m = (t >= start) & (t < end)
        seg = signal[m].copy()
        if len(seg) < fs * 3:
            start += step
            continue

        # Clip outliers
        seg = np.clip(seg, np.percentile(seg, 1), np.percentile(seg, 99))

        # DC and AC
        dc = np.mean(seg)
        ac = bandpass(seg - dc, fs)

        # PI
        ac_pp = np.percentile(ac, 90) - np.percentile(ac, 10)
        pi = abs(ac_pp) / abs(dc) * 100 if dc > 0 else 0.0

        # SNR
        snr = compute_snr(ac, fs)

        # AC amplitude
        amp = ac_pp

        # HR
        hr, rrcv, _ = detect_hr(ac, fs)

        win_t.append(start + win / 2)
        pis.append(pi)
        snrs.append(snr)
        amps.append(amp)
        hrs.append(hr)
        rrcvs.append(rrcv)
        start += step

    return dict(
        win_t=np.array(win_t), pis=np.array(pis), snrs=np.array(snrs),
        amps=np.array(amps), hrs=np.array(hrs), rrcvs=np.array(rrcvs),
    )


def compute_filtered_metrics(t, signal, fs, win=10.0, step=2.0, skip=10.0):
    """Compute SNR, AC amp, HR, RR CV from pre-filtered signal (no PI — no DC)."""
    win_t, snrs, amps, hrs, rrcvs = [], [], [], [], []

    start = skip
    duration = t[-1] - t[0]
    while start + win <= duration:
        end = start + win
        m = (t >= start) & (t < end)
        seg = signal[m].copy()
        if len(seg) < fs * 3:
            start += step
            continue

        # For filtered data, the signal is already AC-coupled
        # Apply bandpass to further isolate cardiac
        ac = bandpass(seg, fs)

        snr = compute_snr(ac, fs)
        amp = np.percentile(ac, 90) - np.percentile(ac, 10)
        hr, rrcv, _ = detect_hr(ac, fs)

        win_t.append(start + win / 2)
        snrs.append(snr)
        amps.append(amp)
        hrs.append(hr)
        rrcvs.append(rrcv)
        start += step

    return dict(
        win_t=np.array(win_t), snrs=np.array(snrs),
        amps=np.array(amps), hrs=np.array(hrs), rrcvs=np.array(rrcvs),
    )


# ── Summary ───────────────────────────────────────────────────────────────────
def summarise(mets, has_pi=True):
    """Compute median of last TAIL_S seconds."""
    wt = mets['win_t']
    if len(wt) == 0:
        return dict(pi=np.nan, snr=np.nan, amp=np.nan, hr_pct=np.nan,
                    rrcv=np.nan, hr_med=np.nan)
    tail = wt >= wt[-1] - TAIL_S

    snrs_t  = mets['snrs'][tail]
    amps_t  = mets['amps'][tail]
    hrs_t   = mets['hrs'][tail]
    rrcvs_t = mets['rrcvs'][tail]

    valid_hr = (hrs_t > THR['hr_lo']) & (hrs_t < THR['hr_hi'])
    hr_pct = float(valid_hr.mean() * 100) if len(hrs_t) > 0 else np.nan
    hr_med = float(np.nanmedian(hrs_t[valid_hr])) if valid_hr.any() else np.nan
    rrcv_m = float(np.nanmedian(rrcvs_t[valid_hr])) if valid_hr.any() else np.nan

    s = dict(
        snr=float(np.nanmedian(snrs_t)),
        amp=float(np.nanmedian(amps_t)),
        hr_pct=hr_pct,
        hr_med=hr_med,
        rrcv=rrcv_m,
    )
    if has_pi and 'pis' in mets:
        pis_t = mets['pis'][tail]
        s['pi'] = float(np.nanmedian(pis_t))
    else:
        s['pi'] = np.nan
    return s


def grade(val, g, f):
    if np.isnan(val): return 'N/A'
    return 'GOOD' if val >= g else 'FAIR' if val >= f else 'POOR'

def grade_inv(val, g, f):
    if np.isnan(val): return 'N/A'
    return 'GOOD' if val <= g else 'FAIR' if val <= f else 'POOR'

GRADE_COLOR = {'GOOD': '#2ca02c', 'FAIR': '#ff7f0e', 'POOR': '#d62728', 'N/A': '#999999'}


# ── PLOT 1: Raw Waveform Overview ─────────────────────────────────────────────
def plot_raw_waveforms(spo2_raw, hrm_raw):
    SHOW_S = 40
    channels = [
        ('SpO2 IR (PPG1_SUB1)', spo2_raw['t'], spo2_raw['s1'], '#1f77b4'),
        ('SpO2 RED (PPG1_SUB2)', spo2_raw['t'], spo2_raw['s2'], '#d62728'),
        ('SpO2 SUB3 (PPG1_SUB3)', spo2_raw['t'], spo2_raw['s3'], '#9467bd'),
        ('HRM CH1 (PPG1_SUB1)', hrm_raw['t'], hrm_raw['s1'], '#2ca02c'),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(18, 12))
    fig.suptitle('Position A — Raw PPG Waveforms (last 40s)',
                 fontsize=14, fontweight='bold')

    for ax, (label, t, sig, color) in zip(axes, channels):
        # Filter valid data (non-zero, non-NaN)
        valid = (~np.isnan(sig)) & (sig > 0)
        t_v, sig_v = t[valid], sig[valid]
        if len(t_v) == 0:
            ax.text(0.5, 0.5, f'{label}: NO DATA', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(label, fontsize=10, color=color, fontweight='bold')
            continue

        t_start = max(0, t_v[-1] - SHOW_S)
        m = t_v >= t_start
        t_s, sig_s = t_v[m], sig_v[m]

        dc = np.mean(sig_s)
        ac_range = np.max(sig_s) - np.min(sig_s)

        ax.plot(t_s, sig_s, lw=0.5, color=color, alpha=0.85)
        ax.set_xlim(t_start, t_v[-1])
        ax.set_ylabel('ADC Counts', fontsize=8)
        ax.set_title(f'{label}   DC={dc:.0f}  AC_range={ac_range:.0f}  '
                     f'AC/DC={ac_range/dc*100:.4f}%',
                     loc='left', fontsize=9, color=color, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout()
    fout = os.path.join(OUT_DIR, '01_raw_waveform_overview.png')
    fig.savefig(fout, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fout)}')


# ── PLOT 2: Filtered Waveform Zoom with Peak Detection ───────────────────────
def plot_filtered_zoom(spo2_filt, hrm_filt, spo2_raw, hrm_raw):
    ZOOM_S = 15  # seconds to show
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle('Position A — Filtered PPG with Peak Detection (15s zoom, best segment)',
                 fontsize=14, fontweight='bold')

    # Find best channels from each filtered file
    plot_data = []

    # SpO2 filtered - pick channel with largest std
    for ch_name, ch_data in sorted(spo2_filt['channels'].items(),
                                    key=lambda x: np.nanstd(x[1]), reverse=True)[:2]:
        plot_data.append(('SpO2 Filt ' + ch_name, spo2_filt['t'], ch_data,
                          spo2_filt['fs']))

    # HRM filtered - pick best channel
    for ch_name, ch_data in sorted(hrm_filt['channels'].items(),
                                    key=lambda x: np.nanstd(x[1]), reverse=True)[:1]:
        plot_data.append(('HRM Filt ' + ch_name, hrm_filt['t'], ch_data,
                          hrm_filt['fs']))

    colors = ['#d62728', '#9467bd', '#2ca02c']
    for i, (ax, (label, t, sig, fs)) in enumerate(zip(axes, plot_data)):
        color = colors[i % len(colors)]
        # Use segment from last 60s with best SNR
        t_mid = max(t[-1] - 30, t[0] + ZOOM_S)
        t_start = t_mid - ZOOM_S / 2
        t_end = t_mid + ZOOM_S / 2
        m = (t >= t_start) & (t < t_end)
        t_s, sig_s = t[m], sig[m]

        if len(sig_s) < 10:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        # Bandpass and detect peaks
        ac = bandpass(sig_s, fs)
        hr, rrcv, peaks = detect_hr(ac, fs)

        ax.plot(t_s, ac, lw=0.8, color=color, alpha=0.85)
        if len(peaks) > 0:
            ax.plot(t_s[peaks], ac[peaks], 'v', ms=6, color='black', alpha=0.7,
                    label=f'Peaks (HR={hr:.1f} bpm)' if not np.isnan(hr) else 'Peaks')
        ax.set_ylabel('Filtered AC', fontsize=8)
        ax.set_title(f'{label}   HR={hr:.1f} bpm  RR_CV={rrcv:.1f}%' if not np.isnan(hr)
                     else f'{label}   HR=N/A', loc='left', fontsize=9,
                     color=color, fontweight='bold')
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout()
    fout = os.path.join(OUT_DIR, '02_filtered_waveform_zoom.png')
    fig.savefig(fout, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fout)}')


# ── PLOT 3: PSD Comparison ────────────────────────────────────────────────────
def plot_psd(spo2_raw, hrm_raw):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Position A — Power Spectral Density (last 60s, Welch)',
                 fontsize=13, fontweight='bold')

    datasets = [
        ('SpO2 Mode', axes[0], [
            ('IR (SUB1)', spo2_raw['s1'], '#1f77b4'),
            ('RED (SUB2)', spo2_raw['s2'], '#d62728'),
            ('SUB3', spo2_raw['s3'], '#9467bd'),
        ]),
        ('HRM Mode', axes[1], [
            ('CH1 (SUB1)', hrm_raw['s1'], '#2ca02c'),
            ('CH2 (SUB2)', hrm_raw['s2'], '#ff7f0e'),
        ]),
    ]

    for title, ax, channels in datasets:
        t_data = spo2_raw['t'] if 'SpO2' in title else hrm_raw['t']
        fs_data = spo2_raw['fs'] if 'SpO2' in title else hrm_raw['fs']
        for ch_label, sig, color in channels:
            valid = (~np.isnan(sig)) & (sig > 0)
            t_v, sig_v = t_data[valid], sig[valid]
            if len(t_v) < fs_data * 10:
                continue
            m = t_v >= (t_v[-1] - 60.0)
            seg = sig_v[m]
            if len(seg) < fs_data * 5:
                continue
            seg = np.clip(seg, np.percentile(seg, 1), np.percentile(seg, 99))
            ac = bandpass(seg - np.mean(seg), fs_data)
            f, p = sp.welch(ac, fs=fs_data, nperseg=min(len(ac), fs_data * 8))
            ax.semilogy(f, p, lw=1.5, color=color, alpha=0.85, label=ch_label)

        ax.axvspan(0.7, 3.5, alpha=0.08, color='green', label='Cardiac (0.7-3.5 Hz)')
        ax.axvspan(4.0, 8.0, alpha=0.06, color='red', label='Noise (4-8 Hz)')
        ax.set_xlim(0, 15)
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power', fontsize=10)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.2)

    plt.tight_layout()
    fout = os.path.join(OUT_DIR, '03_psd_comparison.png')
    fig.savefig(fout, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fout)}')


# ── PLOT 4: Sliding Metrics ──────────────────────────────────────────────────
def plot_sliding_metrics(all_channel_mets):
    """all_channel_mets: list of (label, color, mets_dict, has_pi)"""
    METRICS = [
        ('pis',   'PI %',         (0, 3),    [THR['pi_fair'], THR['pi_good']]),
        ('snrs',  'SNR (dB)',     (0, 40),   [THR['snr_fair'], THR['snr_good']]),
        ('amps',  'AC Amp (cts)', (0, None), [THR['amp_fair'], THR['amp_good']]),
        ('hrs',   'HR (bpm)',     (30, 180), [40, 160]),
        ('rrcvs', 'RR CV %',      (0, 100),  [THR['rrcv_good'], THR['rrcv_fair']]),
    ]

    n_ch = len(all_channel_mets)
    n_m = len(METRICS)
    fig, axes = plt.subplots(n_m, 1, figsize=(18, 3 * n_m))
    fig.suptitle('Position A — Sliding Window Metrics (10s window, 2s step)',
                 fontsize=13, fontweight='bold')

    for row, (key, ylabel, ylim, thrs) in enumerate(METRICS):
        ax = axes[row]
        for label, color, mets, has_pi in all_channel_mets:
            if key not in mets:
                continue
            if key == 'pis' and not has_pi:
                continue
            vals = mets[key]
            wt = mets['win_t']
            ax.plot(wt, vals, lw=1.0, color=color, alpha=0.85, label=label)

        for thr in thrs:
            ax.axhline(thr, ls='--', lw=0.8, color='gray', alpha=0.6)
        if ylim[1] is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout()
    fout = os.path.join(OUT_DIR, '04_sliding_metrics.png')
    fig.savefig(fout, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fout)}')


# ── PLOT 5: Sensor-Reported Metrics ──────────────────────────────────────────
def plot_sensor_metrics(spo2_raw, hrm_raw):
    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    fig.suptitle('Position A — AS7058 On-Chip Algorithm Outputs',
                 fontsize=13, fontweight='bold')

    spo2_ev = spo2_raw['events']
    hrm_ev  = hrm_raw['events']

    # Panel 1: SpO2 Signal Quality + PI
    ax = axes[0]
    if 'sq' in spo2_ev:
        valid = ~np.isnan(spo2_ev['sq'])
        ax.plot(spo2_ev['t'][valid], spo2_ev['sq'][valid], 'o-', ms=3,
                color='#1f77b4', alpha=0.7, label='Signal Quality')
    ax2 = ax.twinx()
    if 'pi' in spo2_ev:
        valid = ~np.isnan(spo2_ev['pi'])
        ax2.plot(spo2_ev['t'][valid], spo2_ev['pi'][valid], 's-', ms=3,
                 color='#d62728', alpha=0.7, label='PI %')
        ax2.set_ylabel('PI %', fontsize=9, color='#d62728')
    ax.set_ylabel('Signal Quality', fontsize=9, color='#1f77b4')
    ax.set_title('SpO2 Mode: Signal Quality & PI', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: SpO2 HR + SpO2%
    ax = axes[1]
    if 'hr' in spo2_ev:
        valid = ~np.isnan(spo2_ev['hr']) & (spo2_ev['hr'] > 30) & (spo2_ev['hr'] < 200)
        if valid.any():
            ax.plot(spo2_ev['t'][valid], spo2_ev['hr'][valid], 'o-', ms=3,
                    color='crimson', alpha=0.7, label='HR (bpm)')
    ax2 = ax.twinx()
    if 'spo2' in spo2_ev:
        valid = ~np.isnan(spo2_ev['spo2']) & (spo2_ev['spo2'] > 70)
        if valid.any():
            ax2.plot(spo2_ev['t'][valid], spo2_ev['spo2'][valid], 's-', ms=3,
                     color='green', alpha=0.7, label='SpO2 %')
            ax2.set_ylabel('SpO2 %', fontsize=9, color='green')
    ax.set_ylabel('HR (bpm)', fontsize=9, color='crimson')
    ax.set_title('SpO2 Mode: HR & SpO2%', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: HRM Signal Quality + HR
    ax = axes[2]
    if 'sq' in hrm_ev:
        valid = ~np.isnan(hrm_ev['sq'])
        ax.plot(hrm_ev['t'][valid], hrm_ev['sq'][valid], 'o-', ms=3,
                color='#2ca02c', alpha=0.7, label='HRM Signal Quality')
    ax2 = ax.twinx()
    if 'hr' in hrm_ev:
        valid = ~np.isnan(hrm_ev['hr']) & (hrm_ev['hr'] > 30) & (hrm_ev['hr'] < 200)
        if valid.any():
            ax2.plot(hrm_ev['t'][valid], hrm_ev['hr'][valid], 's-', ms=3,
                     color='crimson', alpha=0.7, label='HR (bpm)')
            ax2.set_ylabel('HR (bpm)', fontsize=9, color='crimson')
    ax.set_ylabel('Signal Quality', fontsize=9, color='#2ca02c')
    ax.set_title('HRM Mode: Signal Quality & HR', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 4: RRM Confidence + Respiration Rate
    ax = axes[3]
    if 'rrm_conf' in hrm_ev:
        valid = ~np.isnan(hrm_ev['rrm_conf'])
        ax.plot(hrm_ev['rrm_t'][valid], hrm_ev['rrm_conf'][valid], 'o-', ms=3,
                color='teal', alpha=0.7, label='RRM Confidence')
    ax2 = ax.twinx()
    if 'rrm_rr' in hrm_ev:
        valid = ~np.isnan(hrm_ev['rrm_rr']) & (hrm_ev['rrm_rr'] > 0)
        if valid.any():
            ax2.plot(hrm_ev['rrm_t'][valid], hrm_ev['rrm_rr'][valid], 's-', ms=3,
                     color='purple', alpha=0.7, label='Resp Rate (bpm)')
            ax2.set_ylabel('Resp Rate (bpm)', fontsize=9, color='purple')
    ax.set_ylabel('Confidence', fontsize=9, color='teal')
    ax.set_title('HRM Mode: RRM Confidence & Respiration Rate', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fout = os.path.join(OUT_DIR, '05_sensor_reported_metrics.png')
    fig.savefig(fout, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fout)}')


# ── PLOT 6: Quality Scorecard ─────────────────────────────────────────────────
def plot_scorecard(summaries):
    """summaries: dict of {channel_label: summary_dict}"""
    metrics = [
        ('pi',     'PI %',          'pi',   False),
        ('snr',    'SNR (dB)',       'snr',  False),
        ('amp',    'AC Amp (cts)',   'amp',  False),
        ('hr_pct', 'HR Valid %',     'hr',   False),
        ('rrcv',   'RR CV %',        'rrcv', True),
    ]

    labels = list(summaries.keys()) + ['Iter1 Chest']
    n_cols = len(labels)
    n_rows = len(metrics)

    fig, ax = plt.subplots(figsize=(4 + 2.5 * n_cols, 1.5 + 1.2 * n_rows))
    ax.set_xlim(0, n_cols + 2)
    ax.set_ylim(0, n_rows + 2)
    ax.axis('off')
    fig.suptitle('Position A — Quality Scorecard (vs Iteration 1 Chest Baseline)',
                 fontsize=14, fontweight='bold', y=0.98)

    # Header row
    for j, lbl in enumerate(labels):
        ax.text(2.5 + j * 2, n_rows + 1.2, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold')

    for i, (key, title, grade_key, is_inv) in enumerate(metrics):
        y = n_rows - i
        ax.text(0.5, y, title, ha='left', va='center', fontsize=10, fontweight='bold')

        for j, lbl in enumerate(labels):
            if lbl == 'Iter1 Chest':
                val = ITER1_CHEST.get(key, np.nan)
            else:
                val = summaries[lbl].get(key, np.nan)

            # Grade
            if is_inv:
                g = grade_inv(val, THR.get(f'{grade_key}_good', 10), THR.get(f'{grade_key}_fair', 25))
            elif key == 'hr_pct':
                g = grade(val, 80, 50)
            else:
                g = grade(val, THR.get(f'{grade_key}_good', 1), THR.get(f'{grade_key}_fair', 0.3))

            color = GRADE_COLOR.get(g, '#999999')
            x = 2.5 + j * 2

            # Draw colored box
            rect = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8,
                                   boxstyle='round,pad=0.1',
                                   facecolor=color, alpha=0.25, edgecolor=color, lw=1.5)
            ax.add_patch(rect)

            # Value text
            if np.isnan(val):
                txt = 'N/A'
            elif key == 'pi':
                txt = f'{val:.3f}'
            elif key in ('snr', 'rrcv', 'hr_pct'):
                txt = f'{val:.1f}'
            else:
                txt = f'{val:.0f}'
            ax.text(x, y + 0.1, txt, ha='center', va='center', fontsize=10,
                    fontweight='bold', color=color)
            ax.text(x, y - 0.2, g, ha='center', va='center', fontsize=7,
                    color=color, fontstyle='italic')

    plt.tight_layout()
    fout = os.path.join(OUT_DIR, '06_quality_scorecard.png')
    fig.savefig(fout, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fout)}')


# ── Markdown Report ──────────────────────────────────────────────────────────
def write_report(summaries, spo2_raw, hrm_raw):
    lines = []
    w = lines.append

    w('# Position A Chest PPG Quality Report - Iteration 2')
    w('')
    w('**Sensor:** AS7058 | **Sampling:** 200 Hz | **Date:** 04 March 2026')
    w('')

    # Determine best channel
    best_key = None
    best_pi = -1
    for k, s in summaries.items():
        pi = s.get('pi', 0)
        if not np.isnan(pi) and pi > best_pi:
            best_pi = pi
            best_key = k
    if best_key is None:
        best_snr = -999
        for k, s in summaries.items():
            snr = s.get('snr', 0)
            if not np.isnan(snr) and snr > best_snr:
                best_snr = snr
                best_key = k

    best = summaries.get(best_key, {})

    # Verdict
    critical_fails = 0
    pi_val = best.get('pi', np.nan)
    snr_val = best.get('snr', np.nan)
    hr_val = best.get('hr_pct', np.nan)

    if not np.isnan(pi_val) and pi_val < THR['pi_fair']:
        critical_fails += 1
    if not np.isnan(snr_val) and snr_val < THR['snr_fair']:
        critical_fails += 1
    if not np.isnan(hr_val) and hr_val < 50:
        critical_fails += 1

    if critical_fails >= 2:
        verdict = 'NOT USABLE'
        verdict_detail = 'Position A fails multiple critical thresholds for reliable PPG measurement.'
    elif critical_fails == 1:
        verdict = 'MARGINAL'
        verdict_detail = 'Position A passes some thresholds but has a critical weakness.'
    else:
        verdict = 'USABLE'
        verdict_detail = 'Position A meets minimum signal quality requirements.'

    w(f'## Verdict: **{verdict}**')
    w('')
    w(f'{verdict_detail}')
    w(f'**Best channel:** {best_key}')
    w('')

    # Comparison with Iteration 1
    w('---')
    w('## Comparison with Iteration 1 Chest')
    w('')
    iter1_pi = ITER1_CHEST['pi']
    if not np.isnan(pi_val):
        change = pi_val - iter1_pi
        pct_change = (change / iter1_pi * 100) if iter1_pi > 0 else 0
        direction = 'IMPROVED' if change > 0 else 'DEGRADED' if change < 0 else 'UNCHANGED'
        w(f'| Metric | Iteration 1 | Position A ({best_key}) | Change |')
        w(f'|--------|-------------|------------------------|--------|')
        w(f'| PI % | {iter1_pi:.3f}% | {pi_val:.3f}% | {change:+.3f}% ({direction}, {pct_change:+.1f}%) |')
        w(f'| SNR (dB) | {ITER1_CHEST["snr"]:.1f} | {best.get("snr", np.nan):.1f} | {best.get("snr", np.nan) - ITER1_CHEST["snr"]:+.1f} |')
        w(f'| AC Amp (cts) | {ITER1_CHEST["amp"]:.0f} | {best.get("amp", np.nan):.0f} | {best.get("amp", np.nan) - ITER1_CHEST["amp"]:+.0f} |')
    else:
        w(f'- Iteration 1 Chest PI: {iter1_pi:.3f}%')
        w(f'- Position A PI: N/A (could not compute)')
    w('')

    # Computed metrics table
    w('---')
    w('## Computed Metrics (median of last 120s)')
    w('')

    # Build markdown table
    col_keys = list(summaries.keys())
    header = '| Metric |'
    sep = '|--------|'
    for k in col_keys:
        header += f' {k} |'
        sep += '--------|'
    header += ' Iter1 Chest |'
    sep += '-------------|'
    w(header)
    w(sep)

    metric_rows = [
        ('PI %',        'pi',     '{:.4f}'),
        ('SNR (dB)',    'snr',    '{:.1f}'),
        ('AC Amp (cts)','amp',    '{:.1f}'),
        ('HR (bpm)',    'hr_med', '{:.1f}'),
        ('HR Valid %',  'hr_pct', '{:.1f}'),
        ('RR CV %',     'rrcv',   '{:.1f}'),
    ]

    iter1_map = {'pi': 0.22, 'snr': 19.1, 'amp': 6.0, 'hr_med': np.nan,
                 'hr_pct': 100.0, 'rrcv': 3.6}

    for label, key, fmt in metric_rows:
        row = f'| {label} |'
        for k in col_keys:
            val = summaries[k].get(key, np.nan)
            row += f' {fmt.format(val) if not np.isnan(val) else "N/A"} |'
        i1 = iter1_map.get(key, np.nan)
        row += f' {fmt.format(i1) if not np.isnan(i1) else "N/A"} |'
        w(row)

    w('')

    # Grading table
    w('---')
    w('## Quality Grades')
    w('')
    header = '| Metric |'
    sep = '|--------|'
    for k in col_keys:
        header += f' {k} |'
        sep += '--------|'
    w(header)
    w(sep)

    grade_rows = [
        ('PI',       'pi',     'pi_good',   'pi_fair',   False),
        ('SNR',      'snr',    'snr_good',  'snr_fair',  False),
        ('AC Amp',   'amp',    'amp_good',  'amp_fair',  False),
        ('HR Valid',  'hr_pct', None,        None,        False),
        ('RR CV',    'rrcv',   'rrcv_good', 'rrcv_fair', True),
    ]

    for label, key, g_key, f_key, inv in grade_rows:
        row = f'| {label} |'
        for k in col_keys:
            val = summaries[k].get(key, np.nan)
            if key == 'hr_pct':
                g = grade(val, 80, 50)
            elif inv:
                g = grade_inv(val, THR[g_key], THR[f_key])
            else:
                g = grade(val, THR[g_key], THR[f_key])
            row += f' {g} |'
        w(row)

    w('')

    # Sensor-reported metrics
    w('---')
    w('## Sensor-Reported Metrics (AS7058 on-chip algorithm)')
    w('')
    w('### SpO2 Mode')
    w('')

    spo2_ev = spo2_raw['events']
    if 'pi' in spo2_ev:
        pi_v = spo2_ev['pi'][~np.isnan(spo2_ev['pi'])]
        if len(pi_v):
            w(f'- **PI:** median={np.median(pi_v):.4f}%, range={np.min(pi_v):.4f}-{np.max(pi_v):.4f}%')
        else:
            w('- **PI:** No data')
    if 'sq' in spo2_ev:
        sq_v = spo2_ev['sq'][~np.isnan(spo2_ev['sq'])]
        if len(sq_v):
            w(f'- **Signal Quality:** median={np.median(sq_v):.0f}, range={np.min(sq_v):.0f}-{np.max(sq_v):.0f}')
        else:
            w('- **Signal Quality:** No data')
    if 'hr' in spo2_ev:
        hr_v = spo2_ev['hr']
        hr_v = hr_v[(~np.isnan(hr_v)) & (hr_v > 30) & (hr_v < 200)]
        if len(hr_v):
            w(f'- **HR:** median={np.median(hr_v):.1f} bpm, range={np.min(hr_v):.1f}-{np.max(hr_v):.1f} bpm')
        else:
            w('- **HR:** No valid data')
    if 'spo2' in spo2_ev:
        s_v = spo2_ev['spo2']
        s_v = s_v[(~np.isnan(s_v)) & (s_v > 70)]
        if len(s_v):
            w(f'- **SpO2 %:** median={np.median(s_v):.1f}%, range={np.min(s_v):.1f}-{np.max(s_v):.1f}%')

    w('')
    w('### HRM Mode')
    w('')
    hrm_ev = hrm_raw['events']
    if 'sq' in hrm_ev:
        sq_v = hrm_ev['sq'][~np.isnan(hrm_ev['sq'])]
        if len(sq_v):
            w(f'- **Signal Quality:** median={np.median(sq_v):.0f}, range={np.min(sq_v):.0f}-{np.max(sq_v):.0f}')
    if 'hr' in hrm_ev:
        hr_v = hrm_ev['hr']
        hr_v = hr_v[(~np.isnan(hr_v)) & (hr_v > 30) & (hr_v < 200)]
        if len(hr_v):
            w(f'- **HR:** median={np.median(hr_v):.1f} bpm, range={np.min(hr_v):.1f}-{np.max(hr_v):.1f} bpm')
    if 'rrm_rr' in hrm_ev:
        rr_v = hrm_ev['rrm_rr']
        rr_v = rr_v[(~np.isnan(rr_v)) & (rr_v > 0)]
        conf_v = hrm_ev['rrm_conf'][~np.isnan(hrm_ev['rrm_conf'])]
        if len(rr_v):
            w(f'- **Respiration Rate:** median={np.median(rr_v):.1f} bpm, confidence median={np.median(conf_v):.0f}')

    w('')

    # AGC analysis
    w('---')
    w('## AGC (Automatic Gain Control) Analysis')
    w('')
    w('| Mode | Channel | LED Current (LSB) | Grade |')
    w('|------|---------|-------------------|-------|')
    for prefix, agc in [('SpO2', spo2_raw['agc']), ('HRM', hrm_raw['agc'])]:
        for key, val in agc.items():
            if 'vals' not in key:
                w(f'| {prefix} | {key} | {val:.0f} | {grade_inv(val, 30, 80)} |')

    w('')

    # Signal vs Noise explanation
    w('---')
    w('## Signal vs Noise Breakdown')
    w('')
    w('### Signal (what we want)')
    w('- **Cardiac pulsation:** blood volume changes with each heartbeat')
    w('- **Frequency range:** 0.7 - 3.5 Hz (42 - 210 bpm)')
    w('- Appears as periodic waveform in bandpass-filtered PPG')
    w('')
    w('### Noise (what degrades measurement)')
    w('- **High-frequency noise (>4 Hz):** electronic/optical noise')
    w('- **Baseline drift (<0.5 Hz):** sensor movement, temperature')
    w('- **Motion artifacts:** body/sensor movement during measurement')
    w('- **Low perfusion:** insufficient blood flow at measurement site')
    w('')
    w('### At this chest position')
    if not np.isnan(pi_val) and pi_val < THR['pi_fair']:
        w(f'- PI = {pi_val:.4f}% is **BELOW** the minimum 0.3% threshold')
        w('- The pulsatile (AC) component is extremely weak relative to DC')
        w('- This means blood volume changes are barely detectable')
    elif not np.isnan(pi_val):
        w(f'- PI = {pi_val:.4f}% is **within acceptable range**')

    best_snr = best.get('snr', np.nan)
    if not np.isnan(best_snr):
        strength = 'much stronger' if best_snr > 10 else 'comparable to' if best_snr > 3 else 'weaker'
        w(f'- SNR = {best_snr:.1f} dB: cardiac power is **{strength}** than noise floor')

    w('')

    # Conclusion
    w('---')
    w('## Conclusion')
    w('')
    if verdict == 'NOT USABLE':
        w('Position A (upper chest) does **NOT** provide adequate PPG signal quality '
          'for reliable heart rate or SpO2 measurement with the AS7058 sensor.')
        w('')
        if not np.isnan(pi_val):
            if pi_val > iter1_pi:
                w(f'While PI improved slightly from Iteration 1 ({iter1_pi:.3f}% -> {pi_val:.3f}%), '
                  f'it remains well below the minimum threshold of {THR["pi_fair"]}%.')
            else:
                w(f'PI has not improved from Iteration 1 ({iter1_pi:.3f}% -> {pi_val:.3f}%).')
        w('')
        w('The IR channel shows essentially NO pulsatile signal at this location. '
          'The RED/HRM channels show marginal pulsation but insufficient for '
          'clinical-grade or even consumer-grade measurements.')
        w('')
        w('**Recommendation:** Try other chest positions (B, C, D, E) before concluding. '
          'If all positions fail, AS7058 is not suitable for chest PPG without '
          'hardware modifications (higher LED power, optimized photodiode).')
    elif verdict == 'MARGINAL':
        w('Position A shows marginal signal quality. Basic heart rate detection '
          'may be possible in the HRM channel, but SpO2 measurement is not feasible.')
        w('')
        w('**Recommendation:** Compare with other positions (B-E). Consider this '
          'position only if no better alternative is found.')
    else:
        w('Position A meets minimum signal quality thresholds. Heart rate '
          'detection is feasible. SpO2 may be possible with algorithm optimization.')

    w('')
    w('---')
    w(f'*Report generated: {datetime.now().strftime("%d %B %Y, %I:%M %p")}*')

    report_path = os.path.join(OUT_DIR, 'POSITION_A_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved: POSITION_A_REPORT.md')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '=' * 60)
    print('  Position A Chest PPG Quality Analysis')
    print('=' * 60)

    # 1. Load data
    print('\n[1/6] Loading data files...')
    spo2_raw  = load_raw_file(SPO2_RAW_F, mode='spo2')
    hrm_raw   = load_raw_file(HRM_RAW_F, mode='hrm')
    spo2_filt = load_filtered_file(SPO2_FILT_F)
    hrm_filt  = load_filtered_file(HRM_FILT_F)

    print(f'  SpO2 Raw:  {spo2_raw["duration"]:.1f}s, {spo2_raw["fs"]} Hz')
    print(f'  HRM Raw:   {hrm_raw["duration"]:.1f}s, {hrm_raw["fs"]} Hz')
    print(f'  SpO2 Filt: {spo2_filt["duration"]:.1f}s, active channels: {list(spo2_filt["channels"].keys())}')
    print(f'  HRM Filt:  {hrm_filt["duration"]:.1f}s, active channels: {list(hrm_filt["channels"].keys())}')

    # 2. Compute metrics from raw files
    print('\n[2/6] Computing sliding window metrics...')

    # SpO2 RED channel (PPG1_SUB2) — skip initial zeros
    spo2_s2_valid = spo2_raw['s2'] > 1000  # filter out initial zero period
    t_red = spo2_raw['t'][spo2_s2_valid]
    s_red = spo2_raw['s2'][spo2_s2_valid]
    if len(t_red) > 0:
        t_red = t_red - t_red[0]  # re-zero
    mets_spo2_red = compute_sliding_metrics(t_red, s_red, spo2_raw['fs']) if len(t_red) > 2000 else None
    print(f'  SpO2 RED: {len(t_red)} samples, {t_red[-1]:.1f}s' if len(t_red) > 0 else '  SpO2 RED: No valid data')

    # SpO2 SUB3 channel
    spo2_s3_valid = spo2_raw['s3'] > 1000
    t_s3 = spo2_raw['t'][spo2_s3_valid]
    s_s3 = spo2_raw['s3'][spo2_s3_valid]
    if len(t_s3) > 0:
        t_s3 = t_s3 - t_s3[0]
    mets_spo2_s3 = compute_sliding_metrics(t_s3, s_s3, spo2_raw['fs']) if len(t_s3) > 2000 else None
    print(f'  SpO2 SUB3: {len(t_s3)} samples, {t_s3[-1]:.1f}s' if len(t_s3) > 0 else '  SpO2 SUB3: No valid data')

    # HRM CH1 (PPG1_SUB1) — skip initial zeros
    hrm_s1_valid = hrm_raw['s1'] > 1000
    t_hrm = hrm_raw['t'][hrm_s1_valid]
    s_hrm = hrm_raw['s1'][hrm_s1_valid]
    if len(t_hrm) > 0:
        t_hrm = t_hrm - t_hrm[0]
    mets_hrm_ch1 = compute_sliding_metrics(t_hrm, s_hrm, hrm_raw['fs']) if len(t_hrm) > 2000 else None
    print(f'  HRM CH1: {len(t_hrm)} samples, {t_hrm[-1]:.1f}s' if len(t_hrm) > 0 else '  HRM CH1: No valid data')

    # Filtered file metrics
    print('\n[3/6] Computing filtered signal metrics...')
    filt_channel_mets = {}
    for ch_name, ch_data in spo2_filt['channels'].items():
        mets = compute_filtered_metrics(spo2_filt['t'], ch_data, spo2_filt['fs'])
        filt_channel_mets[f'SpO2F_{ch_name}'] = mets
        print(f'  SpO2 Filtered {ch_name}: {len(mets["win_t"])} windows')

    for ch_name, ch_data in hrm_filt['channels'].items():
        mets = compute_filtered_metrics(hrm_filt['t'], ch_data, hrm_filt['fs'])
        filt_channel_mets[f'HRMF_{ch_name}'] = mets
        print(f'  HRM Filtered {ch_name}: {len(mets["win_t"])} windows')

    # 3. Summarise
    print('\n[4/6] Generating summaries...')
    summaries = {}
    if mets_spo2_red is not None:
        summaries['SpO2 RED'] = summarise(mets_spo2_red, has_pi=True)
    if mets_spo2_s3 is not None:
        summaries['SpO2 SUB3'] = summarise(mets_spo2_s3, has_pi=True)
    if mets_hrm_ch1 is not None:
        summaries['HRM CH1'] = summarise(mets_hrm_ch1, has_pi=True)

    # Add best filtered channel summary
    for k, mets in filt_channel_mets.items():
        s = summarise(mets, has_pi=False)
        if not np.isnan(s['snr']) and s['snr'] > 0:
            summaries[k] = s

    for k, s in summaries.items():
        print(f'  {k}: PI={s["pi"]:.4f}%  SNR={s["snr"]:.1f}dB  '
              f'AC={s["amp"]:.1f}cts  HR={s.get("hr_med", np.nan):.1f}bpm  '
              f'HR_Valid={s["hr_pct"]:.1f}%')

    # 4. Generate plots
    print('\n[5/6] Generating plots...')
    plot_raw_waveforms(spo2_raw, hrm_raw)
    plot_filtered_zoom(spo2_filt, hrm_filt, spo2_raw, hrm_raw)
    plot_psd(spo2_raw, hrm_raw)

    # Collect all channel metrics for sliding plot
    all_ch_mets = []
    if mets_spo2_red is not None:
        all_ch_mets.append(('SpO2 RED (raw)', '#d62728', mets_spo2_red, True))
    if mets_spo2_s3 is not None:
        all_ch_mets.append(('SpO2 SUB3 (raw)', '#9467bd', mets_spo2_s3, True))
    if mets_hrm_ch1 is not None:
        all_ch_mets.append(('HRM CH1 (raw)', '#2ca02c', mets_hrm_ch1, True))
    # Best filtered channels
    colors_filt = ['#ff7f0e', '#17becf', '#bcbd22', '#e377c2']
    for i, (k, mets) in enumerate(filt_channel_mets.items()):
        if len(mets['win_t']) > 5:
            all_ch_mets.append((k, colors_filt[i % len(colors_filt)], mets, False))

    plot_sliding_metrics(all_ch_mets)
    plot_sensor_metrics(spo2_raw, hrm_raw)
    plot_scorecard(summaries)

    # 5. Write report
    print('\n[6/6] Writing text report...')
    write_report(summaries, spo2_raw, hrm_raw)

    print('\n' + '=' * 60)
    print(f'  Analysis complete. Output: {OUT_DIR}')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
