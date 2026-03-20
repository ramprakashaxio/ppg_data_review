"""
Chest PPG Signal Quality Analysis - Generalized for any Position
=================================================================
Run:   py -3 position_analysis.py B
       py -3 position_analysis.py C D E    (multiple positions)
       py -3 position_analysis.py ALL      (all positions B-E)

Output: Iteration 2_Test data/Position X/Analysis/

Analyses a given chest placement position from the AS7058 Iteration 2 dataset.
Computes PI, SNR, AC amplitude, HR, and RR CV across SpO2 and HRM modes.
Compares against Iteration 1 chest baseline (PI = 0.22%).
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import scipy.signal as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ITER2_DIR  = os.path.join(SCRIPT_DIR, 'Iteration 2_Test data')

# Quality thresholds (same as Iteration 1)
THR = dict(
    pi_good=1.0,   pi_fair=0.3,
    snr_good=10.0, snr_fair=6.0,
    rrcv_good=10.0, rrcv_fair=25.0,
    amp_good=100,  amp_fair=20,
    hr_lo=40.0,    hr_hi=160.0,
)

ITER1_CHEST = dict(pi=0.22, snr=19.1, amp=6.0, hr_pct=100.0, rrcv=3.6, agc=13.0)
TAIL_S = 120.0

# ── File discovery ────────────────────────────────────────────────────────────
def find_position_dir(pos_letter):
    """Find the folder for a given position, handling the 'Postion A' typo."""
    candidates = [
        os.path.join(ITER2_DIR, f'Position {pos_letter}'),
        os.path.join(ITER2_DIR, f'Postion {pos_letter}'),  # typo variant
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f'Position {pos_letter} folder not found. Tried: {candidates}')


def discover_files(pos_dir):
    """Auto-discover the 4 CSV files (SpO2 raw/filtered, HRM raw/filtered)."""
    spo2_dir = os.path.join(pos_dir, 'SpO2')
    hrm_dir  = os.path.join(pos_dir, 'HRM RRM')

    spo2_csvs = sorted(glob.glob(os.path.join(spo2_dir, '*.csv')))
    hrm_csvs  = sorted(glob.glob(os.path.join(hrm_dir, '*.csv')))

    def split_raw_filt(csvs):
        filt = [f for f in csvs if '_filtered' in os.path.basename(f).lower()]
        raw  = [f for f in csvs if '_filtered' not in os.path.basename(f).lower()]
        return raw[0] if raw else None, filt[0] if filt else None

    # Determine SpO2 vs HRM by checking column headers (handles Position D misnamed files)
    spo2_raw_f, spo2_filt_f = None, None
    hrm_raw_f, hrm_filt_f = None, None

    # First check SpO2 folder
    if spo2_csvs:
        raw, filt = split_raw_filt(spo2_csvs)
        # Verify it's actually SpO2 data by checking headers
        if raw:
            header = pd.read_csv(raw, nrows=0).columns.tolist()
            if any('SPO2' in c.upper() for c in header):
                spo2_raw_f = raw
            else:
                # It's actually HRM data in SpO2 folder (shouldn't happen, but safety)
                pass
        if filt:
            spo2_filt_f = filt

    # Then check HRM folder
    if hrm_csvs:
        raw, filt = split_raw_filt(hrm_csvs)
        if raw:
            header = pd.read_csv(raw, nrows=0).columns.tolist()
            if any('HRM' in c.upper() for c in header):
                hrm_raw_f = raw
            elif any('SPO2' in c.upper() for c in header):
                # Misplaced — this is SpO2 data
                if spo2_raw_f is None:
                    spo2_raw_f = raw
        if filt:
            hrm_filt_f = filt

    # If SpO2 raw wasn't found in SpO2 folder, check if it's misnamed in SpO2 folder
    if spo2_raw_f is None and spo2_csvs:
        raw, _ = split_raw_filt(spo2_csvs)
        if raw:
            spo2_raw_f = raw  # accept whatever is there

    return dict(
        spo2_raw=spo2_raw_f, spo2_filt=spo2_filt_f,
        hrm_raw=hrm_raw_f,   hrm_filt=hrm_filt_f,
    )


# ── Signal processing ─────────────────────────────────────────────────────────
def bandpass(sig, fs, lo=0.5, hi=4.0):
    nyq = fs / 2.0
    b, a = sp.butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype='band')
    return sp.filtfilt(b, a, np.nan_to_num(sig))


def compute_snr(ac_signal, fs):
    f, p = sp.welch(ac_signal - np.mean(ac_signal), fs=fs,
                    nperseg=min(len(ac_signal), int(fs * 8)))
    sig_p   = np.trapz(p[(f >= 0.7) & (f <= 3.5)], f[(f >= 0.7) & (f <= 3.5)])
    noise_p = np.trapz(p[(f >= 4.0) & (f <= 8.0)], f[(f >= 4.0) & (f <= 8.0)])
    return 10 * np.log10(sig_p / noise_p) if (sig_p > 0 and noise_p > 0) else 0.0


def detect_hr(ac_signal, fs):
    dist = int(fs * 0.35)
    thresh = max(0.1 * np.max(np.abs(ac_signal)), 1e-6)
    peaks, _ = sp.find_peaks(ac_signal, distance=dist, height=thresh)
    if len(peaks) >= 4:
        rr = np.diff(peaks) / fs
        valid_rr = rr[(rr > 0.35) & (rr < 1.5)]
        if len(valid_rr) >= 3:
            return 60.0 / np.mean(valid_rr), (np.std(valid_rr) / np.mean(valid_rr)) * 100, peaks
    return np.nan, np.nan, peaks


# ── Data loading ──────────────────────────────────────────────────────────────
def load_raw_file(fpath, mode='spo2'):
    print(f'  Loading {os.path.basename(fpath)} ...')
    raw = pd.read_csv(fpath, low_memory=False)

    ts_col = 'TIMESTAMP [s]'
    t_raw  = pd.to_numeric(raw[ts_col], errors='coerce')
    s1_raw = pd.to_numeric(raw['PPG1_SUB1'], errors='coerce')
    s2_raw = pd.to_numeric(raw['PPG1_SUB2'], errors='coerce')

    ppg_mask = s1_raw.notna()
    t  = t_raw[ppg_mask].values.astype(float)
    s1 = s1_raw[ppg_mask].values.astype(float)
    s2 = s2_raw[ppg_mask].values.astype(float)
    t -= t[0]

    s3 = np.full_like(s1, np.nan)
    if 'PPG1_SUB3' in raw.columns:
        s3_raw = pd.to_numeric(raw['PPG1_SUB3'], errors='coerce')
        s3 = s3_raw[ppg_mask].values.astype(float)

    diffs = np.diff(t)
    fs = round(1.0 / np.median(diffs[diffs > 0]))

    agc_info = {}
    for prefix in ['AGC1', 'AGC2']:
        led_col = f'{prefix}_LED_CURRENT'
        if led_col in raw.columns:
            v = pd.to_numeric(raw[led_col], errors='coerce').dropna()
            if len(v):
                agc_info[f'{prefix}_led'] = float(v.median())
                agc_info[f'{prefix}_led_vals'] = v.values

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
        rrm_col = 'RRM: CONFIDENCE'
        if rrm_col in raw.columns:
            rrm_mask = pd.to_numeric(raw[rrm_col], errors='coerce').notna()
            rrm_df = raw[rrm_mask].copy()
            if len(rrm_df):
                events['rrm_t']    = pd.to_numeric(rrm_df[ts_col], errors='coerce').values - t_raw[ppg_mask].values[0]
                events['rrm_conf'] = pd.to_numeric(rrm_df[rrm_col], errors='coerce').values
                events['rrm_rr']   = pd.to_numeric(rrm_df.get('RRM: RESPIRATION_RATE [bpm]', pd.Series()), errors='coerce').values

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
    print(f'  Loading {os.path.basename(fpath)} ...')
    raw = pd.read_csv(fpath, low_memory=False)
    ts_col = 'TIMESTAMP [s]'
    t = pd.to_numeric(raw[ts_col], errors='coerce').dropna().values.astype(float)
    t -= t[0]

    channels = {}
    for col in raw.columns:
        if col.startswith('PPG') and 'SUB' in col:
            v = pd.to_numeric(raw[col], errors='coerce').values[:len(t)]
            if np.nanstd(v) > 0.1:
                channels[col] = v

    diffs = np.diff(t)
    fs = round(1.0 / np.median(diffs[diffs > 0]))

    return dict(t=t, channels=channels, fs=int(fs),
                duration=float(t[-1] - t[0]), ftype='filtered')


# ── Sliding window metrics ────────────────────────────────────────────────────
def compute_sliding_metrics(t, signal, fs, win=10.0, step=2.0, skip=10.0):
    win_t, pis, snrs, amps, hrs, rrcvs = [], [], [], [], [], []
    start = skip
    duration = t[-1] - t[0]
    while start + win <= duration:
        end = start + win
        m = (t >= start) & (t < end)
        seg = signal[m].copy()
        if len(seg) < fs * 3:
            start += step; continue
        seg = np.clip(seg, np.percentile(seg, 1), np.percentile(seg, 99))
        dc = np.mean(seg)
        ac = bandpass(seg - dc, fs)
        ac_pp = np.percentile(ac, 90) - np.percentile(ac, 10)
        pi = abs(ac_pp) / abs(dc) * 100 if dc > 0 else 0.0
        snr = compute_snr(ac, fs)
        hr, rrcv, _ = detect_hr(ac, fs)
        win_t.append(start + win / 2)
        pis.append(pi); snrs.append(snr); amps.append(ac_pp)
        hrs.append(hr); rrcvs.append(rrcv)
        start += step
    return dict(win_t=np.array(win_t), pis=np.array(pis), snrs=np.array(snrs),
                amps=np.array(amps), hrs=np.array(hrs), rrcvs=np.array(rrcvs))


def compute_filtered_metrics(t, signal, fs, win=10.0, step=2.0, skip=10.0):
    win_t, snrs, amps, hrs, rrcvs = [], [], [], [], []
    start = skip
    duration = t[-1] - t[0]
    while start + win <= duration:
        end = start + win
        m = (t >= start) & (t < end)
        seg = signal[m].copy()
        if len(seg) < fs * 3:
            start += step; continue
        ac = bandpass(seg, fs)
        snr = compute_snr(ac, fs)
        amp = np.percentile(ac, 90) - np.percentile(ac, 10)
        hr, rrcv, _ = detect_hr(ac, fs)
        win_t.append(start + win / 2)
        snrs.append(snr); amps.append(amp); hrs.append(hr); rrcvs.append(rrcv)
        start += step
    return dict(win_t=np.array(win_t), snrs=np.array(snrs),
                amps=np.array(amps), hrs=np.array(hrs), rrcvs=np.array(rrcvs))


# ── Summary ───────────────────────────────────────────────────────────────────
def summarise(mets, has_pi=True):
    wt = mets['win_t']
    if len(wt) == 0:
        return dict(pi=np.nan, snr=np.nan, amp=np.nan, hr_pct=np.nan, rrcv=np.nan, hr_med=np.nan)
    tail = wt >= wt[-1] - TAIL_S
    snrs_t = mets['snrs'][tail]; amps_t = mets['amps'][tail]
    hrs_t = mets['hrs'][tail]; rrcvs_t = mets['rrcvs'][tail]
    valid_hr = (hrs_t > THR['hr_lo']) & (hrs_t < THR['hr_hi'])
    hr_pct = float(valid_hr.mean() * 100) if len(hrs_t) > 0 else np.nan
    hr_med = float(np.nanmedian(hrs_t[valid_hr])) if valid_hr.any() else np.nan
    rrcv_m = float(np.nanmedian(rrcvs_t[valid_hr])) if valid_hr.any() else np.nan
    s = dict(snr=float(np.nanmedian(snrs_t)), amp=float(np.nanmedian(amps_t)),
             hr_pct=hr_pct, hr_med=hr_med, rrcv=rrcv_m)
    if has_pi and 'pis' in mets:
        s['pi'] = float(np.nanmedian(mets['pis'][tail]))
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
def plot_raw_waveforms(spo2_raw, hrm_raw, pos_name, out_dir):
    SHOW_S = 40
    channels = [
        ('SpO2 IR (PPG1_SUB1)', spo2_raw['t'], spo2_raw['s1'], '#1f77b4'),
        ('SpO2 RED (PPG1_SUB2)', spo2_raw['t'], spo2_raw['s2'], '#d62728'),
        ('SpO2 SUB3 (PPG1_SUB3)', spo2_raw['t'], spo2_raw['s3'], '#9467bd'),
        ('HRM CH1 (PPG1_SUB1)', hrm_raw['t'], hrm_raw['s1'], '#2ca02c'),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(18, 12))
    fig.suptitle(f'{pos_name} - Raw PPG Waveforms (last 40s)', fontsize=14, fontweight='bold')
    for ax, (label, t, sig, color) in zip(axes, channels):
        valid = (~np.isnan(sig)) & (sig > 0)
        t_v, sig_v = t[valid], sig[valid]
        if len(t_v) == 0:
            ax.text(0.5, 0.5, f'{label}: NO DATA', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(label, fontsize=10, color=color, fontweight='bold'); continue
        t_start = max(0, t_v[-1] - SHOW_S)
        m = t_v >= t_start; t_s, sig_s = t_v[m], sig_v[m]
        dc = np.mean(sig_s); ac_range = np.max(sig_s) - np.min(sig_s)
        ax.plot(t_s, sig_s, lw=0.5, color=color, alpha=0.85)
        ax.set_xlim(t_start, t_v[-1]); ax.set_ylabel('ADC Counts', fontsize=8)
        ax.set_title(f'{label}   DC={dc:.0f}  AC_range={ac_range:.0f}  AC/DC={ac_range/dc*100:.4f}%',
                     loc='left', fontsize=9, color=color, fontweight='bold')
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_raw_waveform_overview.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  Saved: 01_raw_waveform_overview.png')


# ── PLOT 2: Filtered Waveform Zoom ───────────────────────────────────────────
def plot_filtered_zoom(spo2_filt, hrm_filt, pos_name, out_dir):
    ZOOM_S = 15
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(f'{pos_name} - Filtered PPG with Peak Detection (15s zoom)',
                 fontsize=14, fontweight='bold')
    plot_data = []
    for ch_name, ch_data in sorted(spo2_filt['channels'].items(),
                                    key=lambda x: np.nanstd(x[1]), reverse=True)[:2]:
        plot_data.append(('SpO2 Filt ' + ch_name, spo2_filt['t'], ch_data, spo2_filt['fs']))
    for ch_name, ch_data in sorted(hrm_filt['channels'].items(),
                                    key=lambda x: np.nanstd(x[1]), reverse=True)[:1]:
        plot_data.append(('HRM Filt ' + ch_name, hrm_filt['t'], ch_data, hrm_filt['fs']))
    colors = ['#d62728', '#9467bd', '#2ca02c']
    for i, (ax, (label, t, sig, fs)) in enumerate(zip(axes, plot_data)):
        color = colors[i % len(colors)]
        t_mid = max(t[-1] - 30, t[0] + ZOOM_S)
        t_start, t_end = t_mid - ZOOM_S / 2, t_mid + ZOOM_S / 2
        m = (t >= t_start) & (t < t_end); t_s, sig_s = t[m], sig[m]
        if len(sig_s) < 10:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); continue
        ac = bandpass(sig_s, fs); hr, rrcv, peaks = detect_hr(ac, fs)
        ax.plot(t_s, ac, lw=0.8, color=color, alpha=0.85)
        if len(peaks) > 0:
            ax.plot(t_s[peaks], ac[peaks], 'v', ms=6, color='black', alpha=0.7,
                    label=f'Peaks (HR={hr:.1f} bpm)' if not np.isnan(hr) else 'Peaks')
        ax.set_ylabel('Filtered AC', fontsize=8)
        ax.set_title(f'{label}   HR={hr:.1f} bpm  RR_CV={rrcv:.1f}%' if not np.isnan(hr)
                     else f'{label}   HR=N/A', loc='left', fontsize=9, color=color, fontweight='bold')
        ax.legend(fontsize=8); ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_filtered_waveform_zoom.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  Saved: 02_filtered_waveform_zoom.png')


# ── PLOT 3: PSD Comparison ────────────────────────────────────────────────────
def plot_psd(spo2_raw, hrm_raw, pos_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{pos_name} - Power Spectral Density (last 60s, Welch)', fontsize=13, fontweight='bold')
    datasets = [
        ('SpO2 Mode', axes[0], [('IR (SUB1)', spo2_raw['s1'], '#1f77b4'),
                                 ('RED (SUB2)', spo2_raw['s2'], '#d62728'),
                                 ('SUB3', spo2_raw['s3'], '#9467bd')]),
        ('HRM Mode', axes[1], [('CH1 (SUB1)', hrm_raw['s1'], '#2ca02c'),
                                ('CH2 (SUB2)', hrm_raw['s2'], '#ff7f0e')]),
    ]
    for title, ax, channels in datasets:
        t_data = spo2_raw['t'] if 'SpO2' in title else hrm_raw['t']
        fs_data = spo2_raw['fs'] if 'SpO2' in title else hrm_raw['fs']
        for ch_label, sig, color in channels:
            valid = (~np.isnan(sig)) & (sig > 0)
            t_v, sig_v = t_data[valid], sig[valid]
            if len(t_v) < fs_data * 10: continue
            m = t_v >= (t_v[-1] - 60.0); seg = sig_v[m]
            if len(seg) < fs_data * 5: continue
            seg = np.clip(seg, np.percentile(seg, 1), np.percentile(seg, 99))
            ac = bandpass(seg - np.mean(seg), fs_data)
            f, p = sp.welch(ac, fs=fs_data, nperseg=min(len(ac), fs_data * 8))
            ax.semilogy(f, p, lw=1.5, color=color, alpha=0.85, label=ch_label)
        ax.axvspan(0.7, 3.5, alpha=0.08, color='green', label='Cardiac (0.7-3.5 Hz)')
        ax.axvspan(4.0, 8.0, alpha=0.06, color='red', label='Noise (4-8 Hz)')
        ax.set_xlim(0, 15); ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_psd_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  Saved: 03_psd_comparison.png')


# ── PLOT 4: Sliding Metrics ──────────────────────────────────────────────────
def plot_sliding_metrics(all_channel_mets, pos_name, out_dir):
    METRICS = [
        ('pis', 'PI %', (0, 3), [THR['pi_fair'], THR['pi_good']]),
        ('snrs', 'SNR (dB)', (0, 40), [THR['snr_fair'], THR['snr_good']]),
        ('amps', 'AC Amp (cts)', (0, None), [THR['amp_fair'], THR['amp_good']]),
        ('hrs', 'HR (bpm)', (30, 180), [40, 160]),
        ('rrcvs', 'RR CV %', (0, 100), [THR['rrcv_good'], THR['rrcv_fair']]),
    ]
    n_m = len(METRICS)
    fig, axes = plt.subplots(n_m, 1, figsize=(18, 3 * n_m))
    fig.suptitle(f'{pos_name} - Sliding Window Metrics (10s window, 2s step)',
                 fontsize=13, fontweight='bold')
    for row, (key, ylabel, ylim, thrs) in enumerate(METRICS):
        ax = axes[row]
        for label, color, mets, has_pi in all_channel_mets:
            if key not in mets: continue
            if key == 'pis' and not has_pi: continue
            ax.plot(mets['win_t'], mets[key], lw=1.0, color=color, alpha=0.85, label=label)
        for thr in thrs:
            ax.axhline(thr, ls='--', lw=0.8, color='gray', alpha=0.6)
        if ylim[1] is not None: ax.set_ylim(ylim[0], ylim[1])
        else: ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=9); ax.legend(fontsize=8, loc='upper right')
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_sliding_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  Saved: 04_sliding_metrics.png')


# ── PLOT 5: Sensor-Reported Metrics ──────────────────────────────────────────
def plot_sensor_metrics(spo2_raw, hrm_raw, pos_name, out_dir):
    fig, axes = plt.subplots(4, 1, figsize=(18, 14))
    fig.suptitle(f'{pos_name} - AS7058 On-Chip Algorithm Outputs', fontsize=13, fontweight='bold')
    spo2_ev, hrm_ev = spo2_raw['events'], hrm_raw['events']

    ax = axes[0]
    if 'sq' in spo2_ev:
        valid = ~np.isnan(spo2_ev['sq'])
        ax.plot(spo2_ev['t'][valid], spo2_ev['sq'][valid], 'o-', ms=3, color='#1f77b4', alpha=0.7, label='Signal Quality')
    ax2 = ax.twinx()
    if 'pi' in spo2_ev:
        valid = ~np.isnan(spo2_ev['pi'])
        ax2.plot(spo2_ev['t'][valid], spo2_ev['pi'][valid], 's-', ms=3, color='#d62728', alpha=0.7, label='PI %')
        ax2.set_ylabel('PI %', fontsize=9, color='#d62728')
    ax.set_ylabel('Signal Quality', fontsize=9, color='#1f77b4')
    ax.set_title('SpO2 Mode: Signal Quality & PI', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[1]
    if 'hr' in spo2_ev:
        valid = ~np.isnan(spo2_ev['hr']) & (spo2_ev['hr'] > 30) & (spo2_ev['hr'] < 200)
        if valid.any(): ax.plot(spo2_ev['t'][valid], spo2_ev['hr'][valid], 'o-', ms=3, color='crimson', alpha=0.7, label='HR (bpm)')
    ax2 = ax.twinx()
    if 'spo2' in spo2_ev:
        valid = ~np.isnan(spo2_ev['spo2']) & (spo2_ev['spo2'] > 70)
        if valid.any():
            ax2.plot(spo2_ev['t'][valid], spo2_ev['spo2'][valid], 's-', ms=3, color='green', alpha=0.7, label='SpO2 %')
            ax2.set_ylabel('SpO2 %', fontsize=9, color='green')
    ax.set_ylabel('HR (bpm)', fontsize=9, color='crimson')
    ax.set_title('SpO2 Mode: HR & SpO2%', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[2]
    if 'sq' in hrm_ev:
        valid = ~np.isnan(hrm_ev['sq'])
        ax.plot(hrm_ev['t'][valid], hrm_ev['sq'][valid], 'o-', ms=3, color='#2ca02c', alpha=0.7, label='HRM Signal Quality')
    ax2 = ax.twinx()
    if 'hr' in hrm_ev:
        valid = ~np.isnan(hrm_ev['hr']) & (hrm_ev['hr'] > 30) & (hrm_ev['hr'] < 200)
        if valid.any():
            ax2.plot(hrm_ev['t'][valid], hrm_ev['hr'][valid], 's-', ms=3, color='crimson', alpha=0.7, label='HR (bpm)')
            ax2.set_ylabel('HR (bpm)', fontsize=9, color='crimson')
    ax.set_ylabel('Signal Quality', fontsize=9, color='#2ca02c')
    ax.set_title('HRM Mode: Signal Quality & HR', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[3]
    if 'rrm_conf' in hrm_ev:
        valid = ~np.isnan(hrm_ev['rrm_conf'])
        ax.plot(hrm_ev['rrm_t'][valid], hrm_ev['rrm_conf'][valid], 'o-', ms=3, color='teal', alpha=0.7, label='RRM Confidence')
    ax2 = ax.twinx()
    if 'rrm_rr' in hrm_ev:
        valid = ~np.isnan(hrm_ev['rrm_rr']) & (hrm_ev['rrm_rr'] > 0)
        if valid.any():
            ax2.plot(hrm_ev['rrm_t'][valid], hrm_ev['rrm_rr'][valid], 's-', ms=3, color='purple', alpha=0.7, label='Resp Rate (bpm)')
            ax2.set_ylabel('Resp Rate (bpm)', fontsize=9, color='purple')
    ax.set_ylabel('Confidence', fontsize=9, color='teal')
    ax.set_title('HRM Mode: RRM Confidence & Respiration Rate', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Time (s)', fontsize=9); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_sensor_reported_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  Saved: 05_sensor_reported_metrics.png')


# ── PLOT 6: Quality Scorecard ─────────────────────────────────────────────────
def plot_scorecard(summaries, pos_name, out_dir):
    metrics = [
        ('pi', 'PI %', 'pi', False), ('snr', 'SNR (dB)', 'snr', False),
        ('amp', 'AC Amp (cts)', 'amp', False), ('hr_pct', 'HR Valid %', 'hr', False),
        ('rrcv', 'RR CV %', 'rrcv', True),
    ]
    labels = list(summaries.keys()) + ['Iter1 Chest']
    n_cols, n_rows = len(labels), len(metrics)
    fig, ax = plt.subplots(figsize=(4 + 2.5 * n_cols, 1.5 + 1.2 * n_rows))
    ax.set_xlim(0, n_cols + 2); ax.set_ylim(0, n_rows + 2); ax.axis('off')
    fig.suptitle(f'{pos_name} - Quality Scorecard (vs Iteration 1 Chest Baseline)',
                 fontsize=14, fontweight='bold', y=0.98)
    for j, lbl in enumerate(labels):
        ax.text(2.5 + j * 2, n_rows + 1.2, lbl, ha='center', va='center', fontsize=10, fontweight='bold')
    for i, (key, title, grade_key, is_inv) in enumerate(metrics):
        y = n_rows - i
        ax.text(0.5, y, title, ha='left', va='center', fontsize=10, fontweight='bold')
        for j, lbl in enumerate(labels):
            val = ITER1_CHEST.get(key, np.nan) if lbl == 'Iter1 Chest' else summaries[lbl].get(key, np.nan)
            if is_inv: g = grade_inv(val, THR.get(f'{grade_key}_good', 10), THR.get(f'{grade_key}_fair', 25))
            elif key == 'hr_pct': g = grade(val, 80, 50)
            else: g = grade(val, THR.get(f'{grade_key}_good', 1), THR.get(f'{grade_key}_fair', 0.3))
            color = GRADE_COLOR.get(g, '#999999'); x = 2.5 + j * 2
            rect = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8, boxstyle='round,pad=0.1',
                                   facecolor=color, alpha=0.25, edgecolor=color, lw=1.5)
            ax.add_patch(rect)
            txt = 'N/A' if np.isnan(val) else (f'{val:.3f}' if key == 'pi' else f'{val:.1f}' if key in ('snr','rrcv','hr_pct') else f'{val:.0f}')
            ax.text(x, y + 0.1, txt, ha='center', va='center', fontsize=10, fontweight='bold', color=color)
            ax.text(x, y - 0.2, g, ha='center', va='center', fontsize=7, color=color, fontstyle='italic')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_quality_scorecard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  Saved: 06_quality_scorecard.png')


# ── Markdown Report ──────────────────────────────────────────────────────────
def write_report(summaries, spo2_raw, hrm_raw, pos_name, pos_letter, out_dir):
    lines = []
    w = lines.append

    w(f'# {pos_name} Chest PPG Quality Report - Iteration 2')
    w('')
    w('**Sensor:** AS7058 | **Sampling:** 200 Hz | **Date:** 04 March 2026')
    w('')

    best_key, best_pi = None, -1
    for k, s in summaries.items():
        pi = s.get('pi', 0)
        if not np.isnan(pi) and pi > best_pi: best_pi = pi; best_key = k
    if best_key is None:
        best_snr = -999
        for k, s in summaries.items():
            snr = s.get('snr', 0)
            if not np.isnan(snr) and snr > best_snr: best_snr = snr; best_key = k
    best = summaries.get(best_key, {})

    critical_fails = 0
    pi_val = best.get('pi', np.nan); snr_val = best.get('snr', np.nan); hr_val = best.get('hr_pct', np.nan)
    if not np.isnan(pi_val) and pi_val < THR['pi_fair']: critical_fails += 1
    if not np.isnan(snr_val) and snr_val < THR['snr_fair']: critical_fails += 1
    if not np.isnan(hr_val) and hr_val < 50: critical_fails += 1

    if critical_fails >= 2:
        verdict, verdict_detail = 'NOT USABLE', f'{pos_name} fails multiple critical thresholds for reliable PPG measurement.'
    elif critical_fails == 1:
        verdict, verdict_detail = 'MARGINAL', f'{pos_name} passes some thresholds but has a critical weakness.'
    else:
        verdict, verdict_detail = 'USABLE', f'{pos_name} meets minimum signal quality requirements.'

    w(f'## Verdict: **{verdict}**')
    w(''); w(verdict_detail); w(f'**Best channel:** {best_key}'); w('')

    # Comparison
    w('---'); w('## Comparison with Iteration 1 Chest'); w('')
    iter1_pi = ITER1_CHEST['pi']
    if not np.isnan(pi_val):
        change = pi_val - iter1_pi
        pct_change = (change / iter1_pi * 100) if iter1_pi > 0 else 0
        direction = 'IMPROVED' if change > 0 else 'DEGRADED' if change < 0 else 'UNCHANGED'
        w(f'| Metric | Iteration 1 | {pos_name} ({best_key}) | Change |')
        w('|--------|-------------|------------------------|--------|')
        w(f'| PI % | {iter1_pi:.3f}% | {pi_val:.3f}% | {change:+.3f}% ({direction}, {pct_change:+.1f}%) |')
        w(f'| SNR (dB) | {ITER1_CHEST["snr"]:.1f} | {best.get("snr", np.nan):.1f} | {best.get("snr", np.nan) - ITER1_CHEST["snr"]:+.1f} |')
        w(f'| AC Amp (cts) | {ITER1_CHEST["amp"]:.0f} | {best.get("amp", np.nan):.0f} | {best.get("amp", np.nan) - ITER1_CHEST["amp"]:+.0f} |')
    else:
        w(f'- Iteration 1 Chest PI: {iter1_pi:.3f}%')
        w(f'- {pos_name} PI: N/A (could not compute)')
    w('')

    # Metrics table
    w('---'); w('## Computed Metrics (median of last 120s)'); w('')
    col_keys = list(summaries.keys())
    header = '| Metric |' + ''.join(f' {k} |' for k in col_keys) + ' Iter1 Chest |'
    sep = '|--------|' + ''.join('--------|' for _ in col_keys) + '-------------|'
    w(header); w(sep)
    metric_rows = [('PI %','pi','{:.4f}'), ('SNR (dB)','snr','{:.1f}'), ('AC Amp (cts)','amp','{:.1f}'),
                   ('HR (bpm)','hr_med','{:.1f}'), ('HR Valid %','hr_pct','{:.1f}'), ('RR CV %','rrcv','{:.1f}')]
    iter1_map = {'pi': 0.22, 'snr': 19.1, 'amp': 6.0, 'hr_med': np.nan, 'hr_pct': 100.0, 'rrcv': 3.6}
    for label, key, fmt in metric_rows:
        row = f'| {label} |'
        for k in col_keys:
            val = summaries[k].get(key, np.nan)
            row += f' {fmt.format(val) if not np.isnan(val) else "N/A"} |'
        i1 = iter1_map.get(key, np.nan)
        row += f' {fmt.format(i1) if not np.isnan(i1) else "N/A"} |'
        w(row)
    w('')

    # Grades
    w('---'); w('## Quality Grades'); w('')
    header = '| Metric |' + ''.join(f' {k} |' for k in col_keys)
    sep = '|--------|' + ''.join('--------|' for _ in col_keys)
    w(header); w(sep)
    grade_rows = [('PI','pi','pi_good','pi_fair',False), ('SNR','snr','snr_good','snr_fair',False),
                  ('AC Amp','amp','amp_good','amp_fair',False), ('HR Valid','hr_pct',None,None,False),
                  ('RR CV','rrcv','rrcv_good','rrcv_fair',True)]
    for label, key, g_key, f_key, inv in grade_rows:
        row = f'| {label} |'
        for k in col_keys:
            val = summaries[k].get(key, np.nan)
            if key == 'hr_pct': g = grade(val, 80, 50)
            elif inv: g = grade_inv(val, THR[g_key], THR[f_key])
            else: g = grade(val, THR[g_key], THR[f_key])
            row += f' {g} |'
        w(row)
    w('')

    # Sensor-reported
    w('---'); w('## Sensor-Reported Metrics (AS7058 on-chip algorithm)'); w('')
    w('### SpO2 Mode'); w('')
    spo2_ev = spo2_raw['events']
    if 'pi' in spo2_ev:
        pi_v = spo2_ev['pi'][~np.isnan(spo2_ev['pi'])]
        if len(pi_v): w(f'- **PI:** median={np.median(pi_v):.4f}%, range={np.min(pi_v):.4f}-{np.max(pi_v):.4f}%')
        else: w('- **PI:** No data')
    if 'sq' in spo2_ev:
        sq_v = spo2_ev['sq'][~np.isnan(spo2_ev['sq'])]
        if len(sq_v): w(f'- **Signal Quality:** median={np.median(sq_v):.0f}, range={np.min(sq_v):.0f}-{np.max(sq_v):.0f}')
    if 'hr' in spo2_ev:
        hr_v = spo2_ev['hr']; hr_v = hr_v[(~np.isnan(hr_v)) & (hr_v > 30) & (hr_v < 200)]
        if len(hr_v): w(f'- **HR:** median={np.median(hr_v):.1f} bpm, range={np.min(hr_v):.1f}-{np.max(hr_v):.1f} bpm')
        else: w('- **HR:** No valid data')
    if 'spo2' in spo2_ev:
        s_v = spo2_ev['spo2']; s_v = s_v[(~np.isnan(s_v)) & (s_v > 70)]
        if len(s_v): w(f'- **SpO2 %:** median={np.median(s_v):.1f}%, range={np.min(s_v):.1f}-{np.max(s_v):.1f}%')
    w(''); w('### HRM Mode'); w('')
    hrm_ev = hrm_raw['events']
    if 'sq' in hrm_ev:
        sq_v = hrm_ev['sq'][~np.isnan(hrm_ev['sq'])]
        if len(sq_v): w(f'- **Signal Quality:** median={np.median(sq_v):.0f}, range={np.min(sq_v):.0f}-{np.max(sq_v):.0f}')
    if 'hr' in hrm_ev:
        hr_v = hrm_ev['hr']; hr_v = hr_v[(~np.isnan(hr_v)) & (hr_v > 30) & (hr_v < 200)]
        if len(hr_v): w(f'- **HR:** median={np.median(hr_v):.1f} bpm, range={np.min(hr_v):.1f}-{np.max(hr_v):.1f} bpm')
    if 'rrm_rr' in hrm_ev:
        rr_v = hrm_ev['rrm_rr']; rr_v = rr_v[(~np.isnan(rr_v)) & (rr_v > 0)]
        conf_v = hrm_ev['rrm_conf'][~np.isnan(hrm_ev['rrm_conf'])]
        if len(rr_v): w(f'- **Respiration Rate:** median={np.median(rr_v):.1f} bpm, confidence median={np.median(conf_v):.0f}')
    w('')

    # AGC
    w('---'); w('## AGC (Automatic Gain Control) Analysis'); w('')
    w('| Mode | Channel | LED Current (LSB) | Grade |')
    w('|------|---------|-------------------|-------|')
    for prefix, agc in [('SpO2', spo2_raw['agc']), ('HRM', hrm_raw['agc'])]:
        for key, val in agc.items():
            if 'vals' not in key: w(f'| {prefix} | {key} | {val:.0f} | {grade_inv(val, 30, 80)} |')
    w('')

    # Signal vs Noise
    w('---'); w('## Signal vs Noise Breakdown'); w('')
    w('### Signal (what we want)')
    w('- **Cardiac pulsation:** blood volume changes with each heartbeat')
    w('- **Frequency range:** 0.7 - 3.5 Hz (42 - 210 bpm)')
    w('- Appears as periodic waveform in bandpass-filtered PPG')
    w(''); w('### Noise (what degrades measurement)')
    w('- **High-frequency noise (>4 Hz):** electronic/optical noise')
    w('- **Baseline drift (<0.5 Hz):** sensor movement, temperature')
    w('- **Motion artifacts:** body/sensor movement during measurement')
    w('- **Low perfusion:** insufficient blood flow at measurement site')
    w(''); w(f'### At this chest position')
    if not np.isnan(pi_val) and pi_val < THR['pi_fair']:
        w(f'- PI = {pi_val:.4f}% is **BELOW** the minimum 0.3% threshold')
        w('- The pulsatile (AC) component is extremely weak relative to DC')
    elif not np.isnan(pi_val):
        w(f'- PI = {pi_val:.4f}% is **within acceptable range**')
    best_snr = best.get('snr', np.nan)
    if not np.isnan(best_snr):
        strength = 'much stronger' if best_snr > 10 else 'comparable to' if best_snr > 3 else 'weaker'
        w(f'- SNR = {best_snr:.1f} dB: cardiac power is **{strength}** than noise floor')
    w('')

    # Conclusion
    w('---'); w('## Conclusion'); w('')
    if verdict == 'NOT USABLE':
        w(f'{pos_name} (chest) does **NOT** provide adequate PPG signal quality '
          'for reliable heart rate or SpO2 measurement with the AS7058 sensor.')
        w('')
        if not np.isnan(pi_val):
            if pi_val > iter1_pi:
                w(f'While PI improved from Iteration 1 ({iter1_pi:.3f}% -> {pi_val:.3f}%), '
                  f'it remains below the minimum threshold of {THR["pi_fair"]}%.')
            else:
                w(f'PI has not improved from Iteration 1 ({iter1_pi:.3f}% -> {pi_val:.3f}%).')
        w('')
        w('**Recommendation:** Compare with other chest positions. If all positions fail, '
          'AS7058 is not suitable for chest PPG without hardware modifications.')
    elif verdict == 'MARGINAL':
        w(f'{pos_name} shows marginal signal quality. Basic heart rate detection '
          'may be possible in the HRM channel, but SpO2 measurement is not feasible.')
        w('')
        w('**Recommendation:** Compare with other positions. Consider this position '
          'only if no better alternative is found.')
    else:
        w(f'{pos_name} meets minimum signal quality thresholds. Heart rate '
          'detection is feasible. SpO2 may be possible with algorithm optimization.')
    w('')
    w('---')
    w(f'*Report generated: {datetime.now().strftime("%d %B %Y, %I:%M %p")}*')

    report_path = os.path.join(out_dir, f'POSITION_{pos_letter}_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved: POSITION_{pos_letter}_REPORT.md')


# ── Main analysis for one position ───────────────────────────────────────────
def analyze_position(pos_letter):
    pos_name = f'Position {pos_letter}'
    print('\n' + '=' * 60)
    print(f'  {pos_name} Chest PPG Quality Analysis')
    print('=' * 60)

    pos_dir = find_position_dir(pos_letter)
    out_dir = os.path.join(pos_dir, 'Analysis')
    os.makedirs(out_dir, exist_ok=True)

    files = discover_files(pos_dir)
    print(f'\n  Folder: {pos_dir}')
    for k, v in files.items():
        print(f'  {k}: {os.path.basename(v) if v else "NOT FOUND"}')

    if not files['spo2_raw'] or not files['hrm_raw']:
        print(f'\n  ERROR: Missing required files for {pos_name}. Skipping.')
        return

    # 1. Load
    print('\n[1/6] Loading data files...')
    spo2_raw = load_raw_file(files['spo2_raw'], mode='spo2')
    hrm_raw  = load_raw_file(files['hrm_raw'], mode='hrm')
    spo2_filt = load_filtered_file(files['spo2_filt']) if files['spo2_filt'] else None
    hrm_filt  = load_filtered_file(files['hrm_filt']) if files['hrm_filt'] else None

    print(f'  SpO2 Raw:  {spo2_raw["duration"]:.1f}s, {spo2_raw["fs"]} Hz')
    print(f'  HRM Raw:   {hrm_raw["duration"]:.1f}s, {hrm_raw["fs"]} Hz')
    if spo2_filt: print(f'  SpO2 Filt: {spo2_filt["duration"]:.1f}s, active channels: {list(spo2_filt["channels"].keys())}')
    if hrm_filt: print(f'  HRM Filt:  {hrm_filt["duration"]:.1f}s, active channels: {list(hrm_filt["channels"].keys())}')

    # 2. Raw metrics
    print('\n[2/6] Computing sliding window metrics...')
    def compute_raw_channel(name, t_all, sig_all, fs):
        valid = sig_all > 1000
        t_v, s_v = t_all[valid], sig_all[valid]
        if len(t_v) > 0: t_v = t_v - t_v[0]
        mets = compute_sliding_metrics(t_v, s_v, fs) if len(t_v) > 2000 else None
        print(f'  {name}: {len(t_v)} samples, {t_v[-1]:.1f}s' if len(t_v) > 0 else f'  {name}: No valid data')
        return mets

    mets_spo2_red = compute_raw_channel('SpO2 RED', spo2_raw['t'], spo2_raw['s2'], spo2_raw['fs'])
    mets_spo2_s3  = compute_raw_channel('SpO2 SUB3', spo2_raw['t'], spo2_raw['s3'], spo2_raw['fs'])
    mets_hrm_ch1  = compute_raw_channel('HRM CH1', hrm_raw['t'], hrm_raw['s1'], hrm_raw['fs'])

    # 3. Filtered metrics
    print('\n[3/6] Computing filtered signal metrics...')
    filt_channel_mets = {}
    if spo2_filt:
        for ch_name, ch_data in spo2_filt['channels'].items():
            mets = compute_filtered_metrics(spo2_filt['t'], ch_data, spo2_filt['fs'])
            filt_channel_mets[f'SpO2F_{ch_name}'] = mets
            print(f'  SpO2 Filtered {ch_name}: {len(mets["win_t"])} windows')
    if hrm_filt:
        for ch_name, ch_data in hrm_filt['channels'].items():
            mets = compute_filtered_metrics(hrm_filt['t'], ch_data, hrm_filt['fs'])
            filt_channel_mets[f'HRMF_{ch_name}'] = mets
            print(f'  HRM Filtered {ch_name}: {len(mets["win_t"])} windows')

    # 4. Summarise
    print('\n[4/6] Generating summaries...')
    summaries = {}
    if mets_spo2_red is not None: summaries['SpO2 RED'] = summarise(mets_spo2_red, has_pi=True)
    if mets_spo2_s3 is not None: summaries['SpO2 SUB3'] = summarise(mets_spo2_s3, has_pi=True)
    if mets_hrm_ch1 is not None: summaries['HRM CH1'] = summarise(mets_hrm_ch1, has_pi=True)
    for k, mets in filt_channel_mets.items():
        s = summarise(mets, has_pi=False)
        if not np.isnan(s['snr']) and s['snr'] > 0: summaries[k] = s

    for k, s in summaries.items():
        print(f'  {k}: PI={s["pi"]:.4f}%  SNR={s["snr"]:.1f}dB  AC={s["amp"]:.1f}cts  '
              f'HR={s.get("hr_med", np.nan):.1f}bpm  HR_Valid={s["hr_pct"]:.1f}%')

    # 5. Plots
    print('\n[5/6] Generating plots...')
    plot_raw_waveforms(spo2_raw, hrm_raw, pos_name, out_dir)
    if spo2_filt and hrm_filt:
        plot_filtered_zoom(spo2_filt, hrm_filt, pos_name, out_dir)
    plot_psd(spo2_raw, hrm_raw, pos_name, out_dir)

    all_ch_mets = []
    if mets_spo2_red is not None: all_ch_mets.append(('SpO2 RED (raw)', '#d62728', mets_spo2_red, True))
    if mets_spo2_s3 is not None: all_ch_mets.append(('SpO2 SUB3 (raw)', '#9467bd', mets_spo2_s3, True))
    if mets_hrm_ch1 is not None: all_ch_mets.append(('HRM CH1 (raw)', '#2ca02c', mets_hrm_ch1, True))
    colors_filt = ['#ff7f0e', '#17becf', '#bcbd22', '#e377c2']
    for i, (k, mets) in enumerate(filt_channel_mets.items()):
        if len(mets['win_t']) > 5: all_ch_mets.append((k, colors_filt[i % len(colors_filt)], mets, False))

    plot_sliding_metrics(all_ch_mets, pos_name, out_dir)
    plot_sensor_metrics(spo2_raw, hrm_raw, pos_name, out_dir)
    plot_scorecard(summaries, pos_name, out_dir)

    # 6. Report
    print('\n[6/6] Writing report...')
    write_report(summaries, spo2_raw, hrm_raw, pos_name, pos_letter, out_dir)

    print('\n' + '=' * 60)
    print(f'  {pos_name} analysis complete. Output: {out_dir}')
    print('=' * 60 + '\n')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: py -3 position_analysis.py <position> [position2 ...]')
        print('  position: A, B, C, D, E, or ALL')
        print('  Example:  py -3 position_analysis.py B C D E')
        sys.exit(1)

    positions = sys.argv[1:]
    if 'ALL' in [p.upper() for p in positions]:
        positions = ['A', 'B', 'C', 'D', 'E']
    else:
        positions = [p.upper() for p in positions]

    for pos in positions:
        try:
            analyze_position(pos)
        except Exception as e:
            print(f'\n  ERROR analyzing Position {pos}: {e}')
            import traceback; traceback.print_exc()
            continue
