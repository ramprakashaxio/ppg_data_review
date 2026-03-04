"""
PPG Analysis Script: AS7058 Multi-Site + SP-20 Reference
=========================================================
Analyzes PPG signals from wrist, finger, and chest positions using the AS7058 sensor,
and benchmarks against SP-20 pulse oximeter at the finger position.

Usage:
    python ppg_analysis.py

Output:
    output/01_ppg_waveforms.png       - Time-domain PPG waveforms per location
    output/02_psd_comparison.png      - Power Spectral Density across locations
    output/03_signal_quality.png      - Signal quality metrics over time
    output/04_sensor_outputs.png      - Sensor-computed HR and SpO2 over time
    output/05_hr_algorithm.png        - Custom HR extraction via peak detection
    output/06_spo2_algorithm.png      - Custom SpO2 via R-ratio computation
    output/07_sp20_comparison.png     - SP-20 vs AS7058 Finger V3 comparison
    output/08_location_summary.png    - Box plot summary across body locations
    output/09_motion_artifacts.png    - Accelerometer vs motion artifacts (Wrist)
    Console: Summary table of all datasets with key metrics
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import signal as scipy_signal

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, 'output')

FILES = {
    'Wrist_V1':   os.path.join(BASE, 'AS7058', '01_Wrist_AS7058', 'V1',
                                'wrist_position_nikhil_02.032026.csv'),
    'Wrist_V2':   os.path.join(BASE, 'AS7058', '01_Wrist_AS7058', 'v2',
                                'wrist_position_nikhil_V2_02.032026.csv'),
    'Finger_V1':  os.path.join(BASE, 'AS7058', '02_Finger_AS7058', 'V1',
                                'Finger_position_nikhil_V1_02.032026_2026-03-02_12-06-03.csv'),
    'Finger_V2':  os.path.join(BASE, 'AS7058', '02_Finger_AS7058', 'V2',
                                'Finger_position_nikhil_V2_02.032026_2026-03-02_12-12-17.csv'),
    'Finger_V3':  os.path.join(BASE, 'AS7058', '04_Finger_AS7058_Parallel with SP-20',
                                'Finger_position_nikhil_V3_02.032026_2026-03-02_14-09-26.csv'),
    'Chest_V1':   os.path.join(BASE, 'AS7058', '03_Chest_AS7058', 'V1_wrist algo',
                                'Chest_position_nikhil_V1_02.032026_2026-03-02_13-24-19.csv'),
    'Chest_V2':   os.path.join(BASE, 'AS7058', '03_Chest_AS7058', 'V2_wrist algo',
                                'Chest_position_nikhil_V2_02.032026_2026-03-02_13-36-02.csv'),
    'SP20':       os.path.join(BASE, 'SP-20',
                                'SP-20 _20260302140253.csv'),
}

COLORS = {
    'Wrist':  '#2196F3',   # Blue
    'Finger': '#4CAF50',   # Green
    'Chest':  '#F44336',   # Red
    'SP20':   '#FF9800',   # Orange
}

LOCATION_COLORS = {
    'Wrist_V1':  COLORS['Wrist'],
    'Wrist_V2':  '#64B5F6',
    'Finger_V1': COLORS['Finger'],
    'Finger_V2': '#81C784',
    'Finger_V3': '#2E7D32',
    'Chest_V1':  COLORS['Chest'],
    'Chest_V2':  '#EF9A9A',
}

LOCATION_GROUPS = {
    'Wrist':  ['Wrist_V1',  'Wrist_V2'],
    'Finger': ['Finger_V1', 'Finger_V2', 'Finger_V3'],
    'Chest':  ['Chest_V1',  'Chest_V2'],
}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_as7058(filepath, label):
    """
    Load an AS7058 CSV file and split into PPG rows, ACC rows, and SpO2 event rows.

    Returns a dict with keys:
        ppg         : DataFrame of PPG rows (PPG1_SUB1 not NaN)
        acc         : DataFrame of ACC rows (Schema A only, else empty)
        spo2_events : DataFrame of rows where SIGNAL_QUALITY is not NaN
        fs          : detected PPG sampling rate (Hz)
        schema      : 'A' (Wrist) or 'B' (Finger/Chest)
        label       : identifier string
        duration_s  : total recording duration in seconds
    """
    print(f"  Loading {label} from {os.path.basename(filepath)} ...")
    df = pd.read_csv(filepath, low_memory=False)

    # Rename timestamp column
    ts_col = 'TIMESTAMP [s]'

    # Detect schema
    schema = 'A' if 'ACC_X' in df.columns else 'B'

    # PPG rows: where PPG1_SUB1 is populated (not NaN and not zero at the very start)
    ppg_mask = df['PPG1_SUB1'].notna() & (df['PPG1_SUB1'] != '')
    try:
        ppg_mask = ppg_mask & (df['PPG1_SUB1'].astype(str).str.strip() != '')
    except Exception:
        pass

    ppg_df = df[ppg_mask].copy()
    ppg_df[ts_col] = pd.to_numeric(ppg_df[ts_col], errors='coerce')
    ppg_df['PPG1_SUB1'] = pd.to_numeric(ppg_df['PPG1_SUB1'], errors='coerce')
    ppg_df['PPG1_SUB2'] = pd.to_numeric(ppg_df['PPG1_SUB2'], errors='coerce')
    ppg_df['PPG1_SUB3'] = pd.to_numeric(ppg_df['PPG1_SUB3'], errors='coerce')
    ppg_df = ppg_df.dropna(subset=[ts_col, 'PPG1_SUB1'])
    ppg_df = ppg_df.sort_values(ts_col).reset_index(drop=True)

    # Infer sampling rate from median timestamp diff
    diffs = ppg_df[ts_col].diff().dropna()
    median_diff = diffs.median()
    fs = round(1.0 / median_diff) if median_diff > 0 else 100.0

    # ACC rows (Schema A only)
    if schema == 'A' and 'ACC_X' in df.columns:
        acc_mask = df['ACC_X'].notna() & (df['ACC_X'].astype(str).str.strip() != '')
        acc_df = df[acc_mask].copy()
        acc_df[ts_col] = pd.to_numeric(acc_df[ts_col], errors='coerce')
        for col in ['ACC_X', 'ACC_Y', 'ACC_Z']:
            acc_df[col] = pd.to_numeric(acc_df[col], errors='coerce')
        acc_df = acc_df.dropna(subset=[ts_col, 'ACC_X']).sort_values(ts_col).reset_index(drop=True)
    else:
        acc_df = pd.DataFrame()

    # SpO2 event rows: where SIGNAL_QUALITY is populated
    sq_col = 'SPO2: SIGNAL_QUALITY'
    if sq_col in df.columns:
        sq_mask = df[sq_col].notna() & (df[sq_col].astype(str).str.strip() != '')
        spo2_df = df[sq_mask].copy()
        spo2_df[ts_col] = pd.to_numeric(spo2_df[ts_col], errors='coerce')
        for col in [sq_col, 'SPO2: SPO2 [%]', 'SPO2: HEART_RATE [bpm]',
                    'SPO2: PI [%]', 'SPO2: R']:
            if col in spo2_df.columns:
                spo2_df[col] = pd.to_numeric(spo2_df[col], errors='coerce')
        spo2_df = spo2_df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    else:
        spo2_df = pd.DataFrame()

    duration_s = ppg_df[ts_col].max() - ppg_df[ts_col].min() if len(ppg_df) > 1 else 0

    return {
        'ppg':         ppg_df,
        'acc':         acc_df,
        'spo2_events': spo2_df,
        'fs':          fs,
        'schema':      schema,
        'label':       label,
        'duration_s':  duration_s,
    }


def load_sp20(filepath):
    """
    Load SP-20 reference device CSV.
    Returns DataFrame with columns: time_s, spo2, hr
    """
    print(f"  Loading SP-20 from {os.path.basename(filepath)} ...")
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S %b %d %Y',
                                     errors='coerce')
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)
    df['time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    df['spo2'] = pd.to_numeric(df['Oxygen Level'], errors='coerce')
    df['hr']   = pd.to_numeric(df['Pulse Rate'],   errors='coerce')

    return df[['time_s', 'spo2', 'hr']].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: SIGNAL PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(sig, fs, low=0.5, high=4.0, order=4):
    """Butterworth bandpass filter using zero-phase filtfilt."""
    nyq = fs / 2.0
    lo  = max(low  / nyq, 1e-4)
    hi  = min(high / nyq, 0.9999)
    b, a = scipy_signal.butter(order, [lo, hi], btype='band')
    return scipy_signal.filtfilt(b, a, sig)


def lowpass_filter(sig, fs, cutoff=0.5, order=4):
    """Butterworth lowpass filter."""
    nyq = fs / 2.0
    cut = min(cutoff / nyq, 0.9999)
    b, a = scipy_signal.butter(order, cut, btype='low')
    return scipy_signal.filtfilt(b, a, sig)


def normalize(sig):
    """Min-max normalize a signal to [0, 1]."""
    mn, mx = np.nanmin(sig), np.nanmax(sig)
    if mx == mn:
        return np.zeros_like(sig, dtype=float)
    return (sig - mn) / (mx - mn)


def compute_ac_dc(sig, fs, window_s=4.0):
    """
    Compute AC (pulsatile) and DC (baseline) components using a sliding window.
    AC = std of bandpass-filtered signal in each window
    DC = mean of raw signal in each window
    Returns: (ac_array, dc_array) of same length as sig
    """
    n = len(sig)
    win = int(window_s * fs)
    ac = np.full(n, np.nan)
    dc = np.full(n, np.nan)

    # Bandpass filter the whole signal once
    try:
        sig_bp = bandpass_filter(sig, fs, 0.5, 4.0)
    except Exception:
        sig_bp = sig.copy()

    half = win // 2
    for i in range(n):
        start = max(0, i - half)
        end   = min(n, i + half)
        if end - start < 4:
            continue
        dc[i] = np.mean(sig[start:end])
        ac[i] = np.std(sig_bp[start:end])

    return ac, dc


def compute_r_ratio(ir, red, fs):
    """
    Compute R = (AC_red / DC_red) / (AC_ir / DC_ir) using sliding window.
    Returns R as numpy array (same length as ir).
    """
    ir  = np.asarray(ir, dtype=float)
    red = np.asarray(red, dtype=float)

    ac_ir,  dc_ir  = compute_ac_dc(ir,  fs)
    ac_red, dc_red = compute_ac_dc(red, fs)

    with np.errstate(divide='ignore', invalid='ignore'):
        R = (ac_red / dc_red) / (ac_ir / dc_ir)

    # Clip physically meaningful range
    R = np.where(np.isfinite(R) & (R > 0) & (R < 5), R, np.nan)
    return R


def compute_spo2_from_r(R):
    """
    Empirical formula: SpO2 = 104 - 17 * R
    Valid for reflective PPG. Clips result to [85, 100].
    """
    spo2 = 104.0 - 17.0 * np.asarray(R, dtype=float)
    return np.clip(spo2, 85.0, 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: HR EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_peaks_ppg(ir_filtered, fs, min_hr_bpm=40, max_hr_bpm=200):
    """
    Detect systolic peaks in bandpass-filtered IR PPG signal.
    Returns (peak_indices, rr_intervals_s).
    """
    ir = np.asarray(ir_filtered, dtype=float)
    min_dist = int(fs * 60.0 / max_hr_bpm)
    threshold = 0.3 * np.nanmax(ir)

    peaks, props = scipy_signal.find_peaks(
        ir,
        distance=min_dist,
        height=threshold,
        prominence=0.05 * np.nanmax(ir)
    )

    if len(peaks) < 2:
        return peaks, np.array([])

    rr = np.diff(peaks) / fs
    # Keep only physiologically plausible RR intervals
    valid = (rr >= 60.0 / max_hr_bpm) & (rr <= 60.0 / min_hr_bpm)
    valid_peaks = np.concatenate([[peaks[0]], peaks[1:][valid]])
    valid_rr    = rr[valid]
    return valid_peaks, valid_rr


def compute_hr_timeseries(peak_indices, rr_intervals, timestamps):
    """
    Convert RR intervals to HR timeseries.
    timestamps: array of timestamps for each sample (seconds).
    Returns pandas Series with timestamps of each beat.
    """
    if len(rr_intervals) == 0:
        return pd.Series(dtype=float)
    hr_values = 60.0 / rr_intervals
    beat_times = timestamps[peak_indices[1:len(rr_intervals)+1]]
    return pd.Series(hr_values, index=beat_times)


def compute_hrv_metrics(rr_intervals):
    """
    Compute time-domain HRV metrics from RR intervals (in seconds).
    Returns dict with SDNN, RMSSD, pNN50.
    """
    rr = np.asarray(rr_intervals, dtype=float) * 1000.0  # convert to ms
    if len(rr) < 2:
        return {'SDNN': np.nan, 'RMSSD': np.nan, 'pNN50': np.nan, 'n_beats': len(rr)}

    sdnn  = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50.0) / len(np.diff(rr)) * 100.0

    return {'SDNN': sdnn, 'RMSSD': rmssd, 'pNN50': pnn50, 'n_beats': len(rr)}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: RESPIRATORY RATE
# ─────────────────────────────────────────────────────────────────────────────

def compute_respiratory_rate(ir_raw, fs):
    """
    Estimate respiratory rate from PPG amplitude modulation.
    Uses Hilbert transform to extract envelope, then finds dominant RR frequency.
    Returns estimated RR in breaths/min, or NaN if insufficient data.
    """
    ir = np.asarray(ir_raw, dtype=float)
    if len(ir) < int(fs * 30):  # Need at least 30s
        return np.nan, None, None

    # Step 1: Bandpass filter for HR (0.5-4 Hz)
    ir_hr = bandpass_filter(ir, fs, 0.5, 4.0)

    # Step 2: Amplitude envelope via Hilbert transform
    envelope = np.abs(scipy_signal.hilbert(ir_hr))

    # Step 3: Lowpass to isolate respiratory modulation (<0.5 Hz)
    try:
        env_lp = lowpass_filter(envelope, fs, cutoff=0.5)
    except Exception:
        env_lp = envelope

    # Step 4: Detrend and compute PSD
    env_detrend = scipy_signal.detrend(env_lp)
    freqs, psd = scipy_signal.welch(env_detrend, fs=fs, nperseg=min(len(env_detrend), int(fs*60)))

    # Step 5: Find peak in respiratory band (0.1-0.5 Hz = 6-30 breaths/min)
    rr_mask = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(rr_mask):
        return np.nan, freqs, psd

    rr_freqs = freqs[rr_mask]
    rr_psd   = psd[rr_mask]
    dominant_freq = rr_freqs[np.argmax(rr_psd)]
    rr_bpm = dominant_freq * 60.0

    return rr_bpm, freqs, psd


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5: SIGNAL QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr_from_psd(ir, fs):
    """
    Estimate SNR from Welch PSD.
    Signal band: 0.7–3.5 Hz (HR range)
    Noise band:  3.5–10 Hz (above HR)
    Returns SNR in dB.
    """
    ir = np.asarray(ir, dtype=float)
    nperseg = min(len(ir), int(fs * 8))
    freqs, psd = scipy_signal.welch(ir, fs=fs, nperseg=nperseg)

    sig_mask   = (freqs >= 0.7) & (freqs <= 3.5)
    noise_mask = (freqs >= 3.5) & (freqs <= min(10.0, fs / 2 - 0.1))

    if not np.any(sig_mask) or not np.any(noise_mask):
        return np.nan

    sig_power   = np.trapz(psd[sig_mask],   freqs[sig_mask])
    noise_power = np.trapz(psd[noise_mask], freqs[noise_mask])

    if noise_power <= 0:
        return np.nan

    return 10.0 * np.log10(sig_power / noise_power)


def compute_motion_variance(acc_df, window_s=4.0, fs_acc=10.0):
    """
    Compute motion artifact variance from accelerometer data.
    Returns (time_array, variance_array).
    """
    if acc_df is None or len(acc_df) == 0:
        return np.array([]), np.array([])

    ts_col = 'TIMESTAMP [s]'
    t = acc_df[ts_col].values
    mag = np.sqrt(acc_df['ACC_X'].values**2 +
                  acc_df['ACC_Y'].values**2 +
                  acc_df['ACC_Z'].values**2)

    win = int(window_s * fs_acc)
    half = win // 2
    n = len(mag)
    variance = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - half)
        end   = min(n, i + half)
        if end - start > 1:
            variance[i] = np.std(mag[start:end])

    return t, variance


def get_dominant_hr_freq(ir, fs):
    """Find the dominant frequency in HR band (0.7-3.5 Hz) from Welch PSD."""
    ir = np.asarray(ir, dtype=float)
    nperseg = min(len(ir), int(fs * 8))
    freqs, psd = scipy_signal.welch(ir, fs=fs, nperseg=nperseg)
    hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
    if not np.any(hr_mask):
        return np.nan
    return freqs[hr_mask][np.argmax(psd[hr_mask])]


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_waveforms(datasets):
    """Plot 1: Time-domain PPG waveforms for all datasets."""
    print("\n[Plot 1] PPG Waveforms ...")
    labels = list(datasets.keys())
    n = len(labels)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle('PPG IR Waveforms by Body Location (First 30 Seconds, Normalized)',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, lbl in zip(axes, labels):
        ds = datasets[lbl]
        ppg = ds['ppg']
        ts_col = 'TIMESTAMP [s]'
        t = ppg[ts_col].values
        ir = ppg['PPG1_SUB1'].values.astype(float)

        # Show first 30s
        mask = t <= (t[0] + 30.0)
        t_plot  = t[mask] - t[0]
        ir_plot = normalize(ir[mask])

        color = LOCATION_COLORS.get(lbl, 'gray')
        location = lbl.split('_')[0]

        ax.plot(t_plot, ir_plot, color=color, linewidth=0.6, alpha=0.85)
        ax.set_ylabel('Norm. Amp.', fontsize=8)
        ax.set_title(f'{lbl}  |  Fs={ds["fs"]:.0f} Hz  |  Schema {ds["schema"]}  |  '
                     f'Duration={ds["duration_s"]:.1f}s', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Annotate location
        ax.text(0.99, 0.85, location, transform=ax.transAxes,
                ha='right', va='top', fontsize=9, fontweight='bold',
                color=color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.tight_layout()
    _save(fig, '01_ppg_waveforms.png')


def plot_psd_comparison(datasets):
    """Plot 2: Power Spectral Density comparison across body locations."""
    print("\n[Plot 2] PSD Comparison ...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Power Spectral Density of IR PPG — Body Location Comparison',
                 fontsize=13, fontweight='bold')

    for ax, (location, lbls) in zip(axes, LOCATION_GROUPS.items()):
        ax.set_title(location, fontsize=11, fontweight='bold', color=COLORS[location])

        for lbl in lbls:
            if lbl not in datasets:
                continue
            ds = datasets[lbl]
            ir = ds['ppg']['PPG1_SUB1'].values.astype(float)
            fs = ds['fs']
            nperseg = min(len(ir), int(fs * 8))
            freqs, psd = scipy_signal.welch(ir, fs=fs, nperseg=nperseg)

            # Dominant HR frequency
            dom_f = get_dominant_hr_freq(ir, fs)
            color = LOCATION_COLORS.get(lbl, 'gray')

            ax.semilogy(freqs, psd, color=color, linewidth=1.5,
                        label=f'{lbl} | HR≈{dom_f*60:.0f} bpm')

            if np.isfinite(dom_f):
                ax.axvline(dom_f, color=color, linestyle='--', linewidth=1.0, alpha=0.7)

        ax.set_xlim(0, 5)
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('PSD (a.u.²/Hz)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Mark HR and respiratory bands
        ax.axvspan(0.1, 0.5, alpha=0.07, color='purple', label='RR band')
        ax.axvspan(0.7, 3.5, alpha=0.07, color='green',  label='HR band')
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, '02_psd_comparison.png')


def plot_signal_quality(datasets):
    """Plot 3: Signal quality metrics over time."""
    print("\n[Plot 3] Signal Quality Dashboard ...")

    metrics = [
        ('SPO2: SIGNAL_QUALITY', 'Signal Quality Score'),
        ('SPO2: PI [%]',         'Perfusion Index (%)'),
        ('SPO2: R',              'R-ratio (sensor)'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle('AS7058 Signal Quality Metrics Over Time by Body Location',
                 fontsize=13, fontweight='bold')

    for ax, (col, ylabel) in zip(axes, metrics):
        for lbl, ds in datasets.items():
            spo2_ev = ds.get('spo2_events', pd.DataFrame())
            if len(spo2_ev) == 0 or col not in spo2_ev.columns:
                continue
            ts_col = 'TIMESTAMP [s]'
            t  = spo2_ev[ts_col].values
            v  = pd.to_numeric(spo2_ev[col], errors='coerce').values
            ok = np.isfinite(v)
            if not np.any(ok):
                continue
            color = LOCATION_COLORS.get(lbl, 'gray')
            ax.plot(t[ok] - t[ok][0], v[ok], color=color, linewidth=1.2,
                    label=lbl, alpha=0.8, marker='.', markersize=3)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.tight_layout()
    _save(fig, '03_signal_quality.png')


def plot_sensor_outputs(datasets):
    """Plot 4: Sensor-computed HR and SpO2 over time."""
    print("\n[Plot 4] Sensor HR and SpO2 Outputs ...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('AS7058 Sensor-Computed HR and SpO2 Over Time',
                 fontsize=13, fontweight='bold')

    cols = [('SPO2: HEART_RATE [bpm]', 'Heart Rate (BPM)', [40, 120]),
            ('SPO2: SPO2 [%]',         'SpO2 (%)',         [85, 105])]

    for ax, (col, ylabel, ylim) in zip(axes, cols):
        for lbl, ds in datasets.items():
            spo2_ev = ds.get('spo2_events', pd.DataFrame())
            if len(spo2_ev) == 0 or col not in spo2_ev.columns:
                continue
            ts_col = 'TIMESTAMP [s]'
            t  = spo2_ev['TIMESTAMP [s]'].values
            v  = pd.to_numeric(spo2_ev[col], errors='coerce').values
            ok = np.isfinite(v) & (v > 0)
            if not np.any(ok):
                continue
            color = LOCATION_COLORS.get(lbl, 'gray')
            ax.plot(t[ok] - t[ok][0], v[ok], color=color, linewidth=1.4,
                    label=lbl, alpha=0.85, marker='o', markersize=3)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(ylim)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Time from start (s)', fontsize=9)
    fig.tight_layout()
    _save(fig, '04_sensor_outputs.png')


def plot_hr_algorithm(datasets):
    """Plot 5: Custom HR extraction via peak detection (one subplot per location)."""
    print("\n[Plot 5] HR Algorithm (Peak Detection) ...")

    # Use best representative file per location
    location_reps = {
        'Wrist':  'Wrist_V1',
        'Finger': 'Finger_V1',
        'Chest':  'Chest_V1',
    }

    n = len(location_reps)
    fig = plt.figure(figsize=(16, 5 * n))
    fig.suptitle('Custom HR Extraction via Peak Detection', fontsize=13, fontweight='bold')

    for row_idx, (location, lbl) in enumerate(location_reps.items()):
        if lbl not in datasets:
            continue
        ds = datasets[lbl]
        ppg = ds['ppg']
        fs  = ds['fs']
        ts_col = 'TIMESTAMP [s]'
        t  = ppg[ts_col].values
        ir = ppg['PPG1_SUB1'].values.astype(float)

        # Filter for peak detection
        try:
            ir_filt = bandpass_filter(ir, fs, 0.5, 4.0)
        except Exception:
            ir_filt = ir.copy()

        # Detect peaks
        peaks, rr = detect_peaks_ppg(ir_filt, fs)
        hr_series = compute_hr_timeseries(peaks, rr, t)
        hrv = compute_hrv_metrics(rr)

        # Use first 30s for waveform display
        t_30_mask = t <= (t[0] + 30.0)
        t_disp = t[t_30_mask] - t[0]
        ir_raw_disp  = normalize(ir[t_30_mask])
        ir_filt_disp = normalize(ir_filt[t_30_mask])

        peaks_disp = peaks[t[peaks] <= (t[0] + 30.0)]
        t_peaks = t[peaks_disp] - t[0]

        color = COLORS[location]

        # Row has 3 subplots: raw vs filtered | peaks | HR timeseries
        gs = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.06, right=0.97, hspace=0.5,
                               top=1.0 - row_idx * (1.0 / n) - 0.03,
                               bottom=1.0 - (row_idx + 1) * (1.0 / n) + 0.05)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # A: Raw vs Filtered
        ax1.plot(t_disp, ir_raw_disp,  color='lightgray', linewidth=0.6, label='Raw IR')
        ax1.plot(t_disp, ir_filt_disp, color=color,       linewidth=1.0, label='Bandpass (0.5-4 Hz)')
        ax1.set_title(f'{location}: Raw vs Filtered (0-30s)', fontsize=9)
        ax1.set_xlabel('Time (s)', fontsize=8)
        ax1.set_ylabel('Norm. Amplitude', fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # B: Peaks on filtered
        ax2.plot(t_disp, ir_filt_disp, color=color, linewidth=0.8, alpha=0.7)
        if len(peaks_disp) > 0:
            ax2.plot(t_peaks, normalize(ir_filt[peaks_disp]), 'v',
                     color='red', markersize=5, label=f'{len(peaks_disp)} peaks')
        ax2.set_title(f'{location}: Systolic Peaks Detected', fontsize=9)
        ax2.set_xlabel('Time (s)', fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # C: HR timeseries
        if len(hr_series) > 0:
            ax3.plot(hr_series.index - t[0], hr_series.values, 'o-',
                     color=color, markersize=3, linewidth=1.0, label='Derived HR')

        # Overlay sensor HR
        spo2_ev = ds.get('spo2_events', pd.DataFrame())
        hr_col = 'SPO2: HEART_RATE [bpm]'
        if len(spo2_ev) > 0 and hr_col in spo2_ev.columns:
            t_ev = spo2_ev['TIMESTAMP [s]'].values
            hr_ev = pd.to_numeric(spo2_ev[hr_col], errors='coerce').values
            ok = np.isfinite(hr_ev) & (hr_ev > 0)
            if np.any(ok):
                ax3.scatter(t_ev[ok] - t_ev[ok][0], hr_ev[ok],
                            color='orange', s=15, zorder=5, label='Sensor HR', alpha=0.7)

        ax3.set_title(f'{location}: HR  |  SDNN={hrv["SDNN"]:.1f}ms  RMSSD={hrv["RMSSD"]:.1f}ms',
                      fontsize=9)
        ax3.set_xlabel('Time (s)', fontsize=8)
        ax3.set_ylabel('HR (BPM)', fontsize=8)
        ax3.set_ylim(40, 140)
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

    _save(fig, '05_hr_algorithm.png')


def plot_spo2_algorithm(datasets):
    """Plot 6: Custom SpO2 computation via R-ratio."""
    print("\n[Plot 6] SpO2 Algorithm (R-ratio) ...")

    location_reps = {
        'Wrist':  'Wrist_V1',
        'Finger': 'Finger_V1',
        'Chest':  'Chest_V1',
    }

    n = len(location_reps)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5 * n))
    fig.suptitle('SpO2 Estimation via AC/DC R-Ratio', fontsize=13, fontweight='bold')

    for row, (location, lbl) in enumerate(location_reps.items()):
        if lbl not in datasets:
            for c in range(3):
                axes[row, c].set_visible(False)
            continue

        ds   = datasets[lbl]
        ppg  = ds['ppg']
        fs   = ds['fs']
        ts_col = 'TIMESTAMP [s]'
        t    = ppg[ts_col].values
        ir   = ppg['PPG1_SUB1'].values.astype(float)
        red  = ppg['PPG1_SUB2'].values.astype(float)

        # Remove leading zeros from RED
        red_valid = red != 0
        if np.any(red_valid):
            first_valid = np.argmax(red_valid)
            ir  = ir[first_valid:]
            red = red[first_valid:]
            t   = t[first_valid:]

        # Subsample for speed (use every 5th point for computation)
        step = 5
        ir_s  = ir[::step]
        red_s = red[::step]
        t_s   = t[::step]
        fs_s  = fs / step

        print(f"    Computing R-ratio for {lbl} ({len(ir_s)} points) ...")
        R = compute_r_ratio(ir_s, red_s, fs_s)
        spo2 = compute_spo2_from_r(R)

        # AC/DC for IR and RED
        ac_ir,  dc_ir  = compute_ac_dc(ir_s,  fs_s)
        ac_red, dc_red = compute_ac_dc(red_s, fs_s)

        color = COLORS[location]

        # Subplot A: AC components
        ax = axes[row, 0]
        ax.plot(t_s - t_s[0], ac_ir,  color='#1565C0', linewidth=0.8, label='AC_IR',  alpha=0.85)
        ax.plot(t_s - t_s[0], ac_red, color='#B71C1C', linewidth=0.8, label='AC_RED', alpha=0.85)
        ax.set_title(f'{location}: Computed AC Components', fontsize=9)
        ax.set_ylabel('AC (std of BP signal)', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Subplot B: R-ratio
        ax = axes[row, 1]
        ok_r = np.isfinite(R)
        ax.plot(t_s[ok_r] - t_s[0], R[ok_r], color=color, linewidth=0.8, label='Computed R')

        # Overlay sensor R
        spo2_ev = ds.get('spo2_events', pd.DataFrame())
        if len(spo2_ev) > 0 and 'SPO2: R' in spo2_ev.columns:
            t_ev = spo2_ev['TIMESTAMP [s]'].values
            r_ev = pd.to_numeric(spo2_ev['SPO2: R'], errors='coerce').values
            ok_s = np.isfinite(r_ev) & (r_ev > 0)
            if np.any(ok_s):
                ax.scatter(t_ev[ok_s] - t_ev[ok_s][0], r_ev[ok_s],
                           color='orange', s=10, zorder=5, label='Sensor R', alpha=0.7)

        ax.set_title(f'{location}: R-ratio', fontsize=9)
        ax.set_ylabel('R = (AC_red/DC_red) / (AC_ir/DC_ir)', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylim(0, 3)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Subplot C: SpO2
        ax = axes[row, 2]
        ok_spo2 = np.isfinite(spo2)
        ax.plot(t_s[ok_spo2] - t_s[0], spo2[ok_spo2], color=color,
                linewidth=0.8, label='Computed SpO2 (104-17R)')

        # Overlay sensor SpO2
        if len(spo2_ev) > 0 and 'SPO2: SPO2 [%]' in spo2_ev.columns:
            t_ev = spo2_ev['TIMESTAMP [s]'].values
            sp_ev = pd.to_numeric(spo2_ev['SPO2: SPO2 [%]'], errors='coerce').values
            ok_s = np.isfinite(sp_ev) & (sp_ev > 0)
            if np.any(ok_s):
                ax.scatter(t_ev[ok_s] - t_ev[ok_s][0], sp_ev[ok_s],
                           color='orange', s=10, zorder=5, label='Sensor SpO2', alpha=0.7)

        ax.set_title(f'{location}: SpO2 Estimate', fontsize=9)
        ax.set_ylabel('SpO2 (%)', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylim(85, 102)
        ax.axhline(95, color='red', linestyle=':', linewidth=0.8, alpha=0.5, label='95% threshold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, '06_spo2_algorithm.png')


def plot_sp20_comparison(finger_v3, sp20_df):
    """Plot 7: SP-20 vs AS7058 Finger V3 comparison with Bland-Altman."""
    print("\n[Plot 7] SP-20 vs AS7058 Comparison ...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SP-20 Reference vs AS7058 Finger V3 — Parallel Measurement Comparison',
                 fontsize=13, fontweight='bold')

    # Extract sensor HR and SpO2 from Finger V3
    spo2_ev = finger_v3.get('spo2_events', pd.DataFrame())

    as7058_hr   = pd.Series(dtype=float)
    as7058_spo2 = pd.Series(dtype=float)

    if len(spo2_ev) > 0:
        ts_col = 'TIMESTAMP [s]'
        t_ev = spo2_ev[ts_col].values - spo2_ev[ts_col].values[0]

        hr_col = 'SPO2: HEART_RATE [bpm]'
        sp_col = 'SPO2: SPO2 [%]'

        if hr_col in spo2_ev.columns:
            hr_vals = pd.to_numeric(spo2_ev[hr_col], errors='coerce').values
            ok = np.isfinite(hr_vals) & (hr_vals > 0)
            as7058_hr = pd.Series(hr_vals[ok], index=t_ev[ok])

        if sp_col in spo2_ev.columns:
            sp_vals = pd.to_numeric(spo2_ev[sp_col], errors='coerce').values
            ok = np.isfinite(sp_vals) & (sp_vals > 0)
            as7058_spo2 = pd.Series(sp_vals[ok], index=t_ev[ok])

    sp20_hr   = pd.Series(sp20_df['hr'].values,   index=sp20_df['time_s'].values)
    sp20_spo2 = pd.Series(sp20_df['spo2'].values, index=sp20_df['time_s'].values)

    # Panel A: HR timeseries
    ax = axes[0, 0]
    ax.plot(sp20_hr.index,   sp20_hr.values,   color=COLORS['SP20'],   linewidth=1.5,
            marker='o', markersize=3, label='SP-20 HR')
    if len(as7058_hr) > 0:
        ax.plot(as7058_hr.index, as7058_hr.values, color=COLORS['Finger'], linewidth=1.5,
                marker='s', markersize=3, label='AS7058 Finger V3 HR')
    ax.set_title('Heart Rate Comparison', fontsize=10, fontweight='bold')
    ax.set_xlabel('Elapsed Time (s)', fontsize=9)
    ax.set_ylabel('HR (BPM)', fontsize=9)
    ax.set_ylim(40, 140)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: SpO2 timeseries
    ax = axes[0, 1]
    ax.plot(sp20_spo2.index,   sp20_spo2.values,   color=COLORS['SP20'],   linewidth=1.5,
            marker='o', markersize=3, label='SP-20 SpO2')
    if len(as7058_spo2) > 0:
        ax.plot(as7058_spo2.index, as7058_spo2.values, color=COLORS['Finger'], linewidth=1.5,
                marker='s', markersize=3, label='AS7058 Finger V3 SpO2')
    ax.set_title('SpO2 Comparison', fontsize=10, fontweight='bold')
    ax.set_xlabel('Elapsed Time (s)', fontsize=9)
    ax.set_ylabel('SpO2 (%)', fontsize=9)
    ax.set_ylim(85, 105)
    ax.axhline(95, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Align and interpolate for Bland-Altman (use common time axis)
    def align_series(s1, s2, kind='linear'):
        """Interpolate s2 at s1's timestamps."""
        if len(s1) == 0 or len(s2) == 0:
            return np.array([]), np.array([])
        from scipy.interpolate import interp1d
        t_common_min = max(s1.index[0],  s2.index[0])
        t_common_max = min(s1.index[-1], s2.index[-1])
        if t_common_max <= t_common_min:
            return np.array([]), np.array([])
        t_ref = s1.index[(s1.index >= t_common_min) & (s1.index <= t_common_max)]
        if len(t_ref) < 2:
            return np.array([]), np.array([])
        f2 = interp1d(s2.index, s2.values, kind=kind, bounds_error=False, fill_value=np.nan)
        v1 = s1[t_ref].values
        v2 = f2(t_ref)
        ok = np.isfinite(v1) & np.isfinite(v2)
        return v1[ok], v2[ok]

    # Panel C: Bland-Altman for HR
    ax = axes[1, 0]
    v_sp20, v_as = align_series(sp20_hr, as7058_hr)
    if len(v_sp20) > 2:
        means = (v_sp20 + v_as) / 2
        diffs = v_as - v_sp20
        bias  = np.mean(diffs)
        loa   = 1.96 * np.std(diffs)
        ax.scatter(means, diffs, color=COLORS['Finger'], alpha=0.6, s=20)
        ax.axhline(bias, color='red',    linewidth=1.5, linestyle='-',  label=f'Bias={bias:.1f}')
        ax.axhline(bias + loa, color='orange', linewidth=1.2, linestyle='--',
                   label=f'+1.96σ={bias+loa:.1f}')
        ax.axhline(bias - loa, color='orange', linewidth=1.2, linestyle='--',
                   label=f'-1.96σ={bias-loa:.1f}')
        ax.set_title('Bland-Altman: HR (AS7058 - SP-20)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Mean HR (BPM)', fontsize=9)
        ax.set_ylabel('Difference (BPM)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel D: SpO2 scatter
        ax = axes[1, 1]
        v_sp20_s, v_as_s = align_series(sp20_spo2, as7058_spo2)
        if len(v_sp20_s) > 2:
            corr = np.corrcoef(v_sp20_s, v_as_s)[0, 1]
            ax.scatter(v_sp20_s, v_as_s, color=COLORS['Finger'], alpha=0.6, s=20)
            min_v = min(v_sp20_s.min(), v_as_s.min()) - 1
            max_v = max(v_sp20_s.max(), v_as_s.max()) + 1
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.0, label='Identity')
            ax.set_title(f'SpO2 Scatter  |  r={corr:.3f}', fontsize=10, fontweight='bold')
            ax.set_xlabel('SP-20 SpO2 (%)', fontsize=9)
            ax.set_ylabel('AS7058 SpO2 (%)', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient overlapping SpO2 data\n(sensor SpO2 intermittent)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10,
                    color='gray')
            ax.set_title('SpO2 Scatter (insufficient data)', fontsize=10)
    else:
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.text(0.5, 0.5,
                    'Insufficient overlapping HR data\n(AS7058 sensor HR intermittent output)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
        axes[1, 0].set_title('Bland-Altman HR (insufficient data)', fontsize=10)
        axes[1, 1].set_title('SpO2 Scatter (insufficient data)',    fontsize=10)

    fig.tight_layout()
    _save(fig, '07_sp20_comparison.png')


def plot_location_summary(datasets, snr_dict, hrv_dict):
    """Plot 8: Box plots summarizing metrics by body location."""
    print("\n[Plot 8] Location Summary ...")

    # Build summary DataFrame
    rows = []
    for lbl, ds in datasets.items():
        location = lbl.split('_')[0]
        spo2_ev  = ds.get('spo2_events', pd.DataFrame())

        snr = snr_dict.get(lbl, np.nan)

        hr_vals  = []
        spo2_vals = []
        pi_vals  = []
        sq_vals  = []

        if len(spo2_ev) > 0:
            if 'SPO2: HEART_RATE [bpm]' in spo2_ev.columns:
                v = pd.to_numeric(spo2_ev['SPO2: HEART_RATE [bpm]'], errors='coerce').dropna()
                hr_vals = v[v > 0].tolist()
            if 'SPO2: SPO2 [%]' in spo2_ev.columns:
                v = pd.to_numeric(spo2_ev['SPO2: SPO2 [%]'], errors='coerce').dropna()
                spo2_vals = v[v > 0].tolist()
            if 'SPO2: PI [%]' in spo2_ev.columns:
                v = pd.to_numeric(spo2_ev['SPO2: PI [%]'], errors='coerce').dropna()
                pi_vals = v[v > 0].tolist()
            if 'SPO2: SIGNAL_QUALITY' in spo2_ev.columns:
                v = pd.to_numeric(spo2_ev['SPO2: SIGNAL_QUALITY'], errors='coerce').dropna()
                sq_vals = v.tolist()

        for hr in hr_vals:
            rows.append({'Location': location, 'Metric': 'HR (BPM)',         'Value': hr})
        for sp in spo2_vals:
            rows.append({'Location': location, 'Metric': 'SpO2 (%)',          'Value': sp})
        for pi in pi_vals:
            rows.append({'Location': location, 'Metric': 'PI (%)',            'Value': pi})
        for sq in sq_vals:
            rows.append({'Location': location, 'Metric': 'Signal Quality',    'Value': sq})
        if np.isfinite(snr):
            rows.append({'Location': location, 'Metric': 'SNR (dB)',          'Value': snr})

    df = pd.DataFrame(rows)

    metrics = ['SNR (dB)', 'PI (%)', 'HR (BPM)', 'SpO2 (%)']
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    fig.suptitle('Signal Quality Summary by Body Location', fontsize=13, fontweight='bold')

    palette = {'Wrist': COLORS['Wrist'], 'Finger': COLORS['Finger'], 'Chest': COLORS['Chest']}

    for ax, metric in zip(axes, metrics):
        sub = df[df['Metric'] == metric]
        if len(sub) == 0:
            ax.text(0.5, 0.5, f'No data\nfor {metric}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(metric, fontsize=10)
            continue
        sns.boxplot(data=sub, x='Location', y='Value', palette=palette, ax=ax,
                    order=['Wrist', 'Finger', 'Chest'])
        sns.stripplot(data=sub, x='Location', y='Value', color='black', size=3,
                      alpha=0.5, jitter=True, ax=ax, order=['Wrist', 'Finger', 'Chest'])
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=8)

        # Add thresholds
        if metric == 'SNR (dB)':
            ax.axhline(6, color='red', linestyle='--', linewidth=0.8, label='6 dB threshold')
            ax.legend(fontsize=7)
        elif metric == 'SpO2 (%)':
            ax.axhline(95, color='red', linestyle='--', linewidth=0.8, label='95% threshold')
            ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, '08_location_summary.png')


def plot_motion_artifacts(wrist_v1):
    """Plot 9: Accelerometer vs motion artifacts (Wrist Schema A only)."""
    print("\n[Plot 9] Motion Artifacts ...")

    acc_df  = wrist_v1.get('acc', pd.DataFrame())
    ppg_df  = wrist_v1['ppg']
    spo2_ev = wrist_v1.get('spo2_events', pd.DataFrame())
    ts_col  = 'TIMESTAMP [s]'

    if len(acc_df) == 0:
        print("    No accelerometer data found (Schema B file). Skipping plot 9.")
        return

    t_acc, motion_var = compute_motion_variance(acc_df)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Wrist V1: Motion Artifacts vs PPG Signal Quality',
                 fontsize=13, fontweight='bold')

    # ACC magnitude
    acc_mag = np.sqrt(acc_df['ACC_X'].values**2 +
                      acc_df['ACC_Y'].values**2 +
                      acc_df['ACC_Z'].values**2)
    ax = axes[0]
    ax.plot(acc_df[ts_col].values - acc_df[ts_col].values[0], acc_mag,
            color='purple', linewidth=0.7, alpha=0.8)
    ax.fill_between(t_acc - acc_df[ts_col].values[0], motion_var,
                    color='purple', alpha=0.2, label='Motion σ')
    ax.set_ylabel('ACC Magnitude (LSB)', fontsize=9)
    ax.set_title('Accelerometer Magnitude', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # PPG IR
    t_ppg = ppg_df[ts_col].values - ppg_df[ts_col].values[0]
    ir     = ppg_df['PPG1_SUB1'].values.astype(float)
    ax = axes[1]
    ax.plot(t_ppg, normalize(ir), color=COLORS['Wrist'], linewidth=0.5, alpha=0.8)
    ax.set_ylabel('Norm. IR PPG', fontsize=9)
    ax.set_title('PPG IR Signal', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Signal Quality
    ax = axes[2]
    if len(spo2_ev) > 0 and 'SPO2: SIGNAL_QUALITY' in spo2_ev.columns:
        t_sq = spo2_ev[ts_col].values - spo2_ev[ts_col].values[0]
        sq   = pd.to_numeric(spo2_ev['SPO2: SIGNAL_QUALITY'], errors='coerce').values
        ok   = np.isfinite(sq)
        ax.plot(t_sq[ok], sq[ok], color='darkgreen', linewidth=1.0,
                marker='o', markersize=3, label='Signal Quality')
        ax.set_ylim(0, max(sq[ok].max() * 1.1, 10) if np.any(ok) else 10)
    ax.set_ylabel('Signal Quality', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_title('Sensor Signal Quality Score', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, '09_motion_artifacts.png')


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7: SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(datasets, snr_dict, hrv_dict, rr_dict):
    """Print a formatted summary table of all datasets with key metrics."""

    header = (f"{'Label':<14} {'Loc':<8} {'Fs(Hz)':<8} {'Duration':<10} "
              f"{'SNR(dB)':<10} {'Med.HR':<9} {'Med.SpO2':<10} "
              f"{'Med.PI%':<9} {'SDNN(ms)':<10} {'RR(bpm)':<9}")
    sep = '-' * len(header)

    print('\n' + '=' * len(header))
    print(' DATASET SUMMARY: AS7058 Multi-Site PPG Analysis')
    print('=' * len(header))
    print(header)
    print(sep)

    for lbl, ds in datasets.items():
        location = lbl.split('_')[0]
        fs       = ds['fs']
        dur      = ds['duration_s']
        snr      = snr_dict.get(lbl, np.nan)
        hrv      = hrv_dict.get(lbl, {})
        rr_bpm   = rr_dict.get(lbl, np.nan)

        spo2_ev = ds.get('spo2_events', pd.DataFrame())
        med_hr   = np.nan
        med_spo2 = np.nan
        med_pi   = np.nan

        if len(spo2_ev) > 0:
            for col, var in [('SPO2: HEART_RATE [bpm]', 'med_hr'),
                             ('SPO2: SPO2 [%]',         'med_spo2'),
                             ('SPO2: PI [%]',           'med_pi')]:
                if col in spo2_ev.columns:
                    v = pd.to_numeric(spo2_ev[col], errors='coerce').dropna()
                    v = v[v > 0]
                    if len(v) > 0:
                        locals()[var]
                        if var == 'med_hr':   med_hr   = np.median(v)
                        if var == 'med_spo2': med_spo2 = np.median(v)
                        if var == 'med_pi':   med_pi   = np.median(v)

        sdnn = hrv.get('SDNN', np.nan)

        def fmt(v, d=1):
            return f'{v:.{d}f}' if np.isfinite(v) else 'N/A'

        print(f"{lbl:<14} {location:<8} {fs:<8.0f} {dur:<10.1f} "
              f"{fmt(snr):<10} {fmt(med_hr):<9} {fmt(med_spo2):<10} "
              f"{fmt(med_pi):<9} {fmt(sdnn):<10} {fmt(rr_bpm):<9}")

    print(sep)

    print('\n KEY ASSESSMENT:')
    print('  Finger:  Expected best SNR. Strongest pulsatile signal.')
    print('  Wrist:   Moderate SNR. Motion artifacts visible in ACC data.')
    print('  Chest:   Wrist algorithm applied — may see lower PI and SNR.')
    print('           If Chest SNR > 6 dB and PI > 0.3%, viable for POC stage.')
    print('  SP-20:   1 Hz reference. Check Plot 7 Bland-Altman bias < 3 BPM for HR.')
    print('  SpO2:    Empirical formula (104-17R) is calibrated for fingertip transmission.')
    print('           Calibration coefficients may need adjustment for wrist/chest (reflective).')
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*60)
    print(' PPG Analysis: AS7058 + SP-20')
    print('='*60)

    # ── Load all AS7058 datasets ──
    print('\n[1/7] Loading AS7058 data files ...')
    datasets = {}
    for lbl, fpath in FILES.items():
        if lbl == 'SP20':
            continue
        if not os.path.isfile(fpath):
            print(f"  WARNING: File not found — {fpath}")
            continue
        try:
            datasets[lbl] = load_as7058(fpath, lbl)
        except Exception as e:
            print(f"  ERROR loading {lbl}: {e}")

    # ── Load SP-20 ──
    print('\n[2/7] Loading SP-20 data ...')
    sp20_df = pd.DataFrame()
    sp20_fpath = FILES['SP20']
    if os.path.isfile(sp20_fpath):
        try:
            sp20_df = load_sp20(sp20_fpath)
            print(f"  SP-20: {len(sp20_df)} readings, duration={sp20_df['time_s'].max():.0f}s")
        except Exception as e:
            print(f"  ERROR loading SP-20: {e}")
    else:
        print(f"  WARNING: SP-20 file not found — {sp20_fpath}")

    # ── Compute SNR and HRV for all datasets ──
    print('\n[3/7] Computing SNR, HR, HRV metrics ...')
    snr_dict = {}
    hrv_dict = {}
    rr_dict  = {}

    for lbl, ds in datasets.items():
        print(f"  Processing {lbl} ...")
        ppg = ds['ppg']
        fs  = ds['fs']
        ir  = ppg['PPG1_SUB1'].values.astype(float)

        # SNR
        snr_dict[lbl] = compute_snr_from_psd(ir, fs)

        # HR and HRV
        try:
            ir_filt = bandpass_filter(ir, fs, 0.5, 4.0)
            peaks, rr = detect_peaks_ppg(ir_filt, fs)
            hrv_dict[lbl] = compute_hrv_metrics(rr)
        except Exception as e:
            hrv_dict[lbl] = {'SDNN': np.nan, 'RMSSD': np.nan, 'pNN50': np.nan}
            print(f"    HRV error: {e}")

        # Respiratory rate
        try:
            rr_bpm, _, _ = compute_respiratory_rate(ir, fs)
            rr_dict[lbl] = rr_bpm
        except Exception as e:
            rr_dict[lbl] = np.nan

    # ── Generate all 9 plots ──
    print('\n[4/7] Generating visualizations ...')

    plot_waveforms(datasets)
    plot_psd_comparison(datasets)
    plot_signal_quality(datasets)
    plot_sensor_outputs(datasets)
    plot_hr_algorithm(datasets)
    plot_spo2_algorithm(datasets)

    # SP-20 comparison
    finger_v3 = datasets.get('Finger_V3', None)
    if finger_v3 is not None and len(sp20_df) > 0:
        plot_sp20_comparison(finger_v3, sp20_df)
    else:
        print("\n[Plot 7] Skipping SP-20 comparison (missing data).")

    plot_location_summary(datasets, snr_dict, hrv_dict)

    wrist_v1 = datasets.get('Wrist_V1', None)
    if wrist_v1 is not None:
        plot_motion_artifacts(wrist_v1)
    else:
        print("\n[Plot 9] Skipping motion artifact plot (Wrist_V1 not loaded).")

    # ── Print summary table ──
    print('\n[5/7] Printing summary table ...')
    print_summary_table(datasets, snr_dict, hrv_dict, rr_dict)

    print(f"\n All plots saved to: {OUTPUT_DIR}")
    print(' Done.\n')


if __name__ == '__main__':
    main()
