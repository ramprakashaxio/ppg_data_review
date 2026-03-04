"""
PPG Quality Analysis — Per-Dataset Detailed View
==================================================
Focused analysis of each AS7058 dataset + SP-20 reference.

What this script generates:
  A — Per-dataset channel plots (IR, RED, 3rd channel — raw + AC-filtered)
  B — Per-dataset quality scorecard (SNR, PI%, AC/DC ratio, AGC current, stability)
  C — Cross-location comparison (AC waveform amplitude vs body site)
  D — SP-20 trend (HR + SpO2 over 6 minutes)
  E — AGC current vs signal quality (sensor self-calibration view)

Axis conventions used throughout:
  X axis: Always "Time (seconds)" — elapsed from 0
  Y axis: Depends on view:
    Raw signal   → ADC Counts (LSB) — integer values from 18-bit ADC
    AC-filtered  → ΔADCounts (AC component, mean-removed)
    Normalised   → Amplitude (0–1)
    PSD          → Power Spectral Density (ADC² / Hz), log scale
    HR           → Heart Rate (BPM)
    SpO2         → SpO2 (%)
    AGC current  → LED Current (mA)  [each LSB = 0.4 mA typically for AS7058]

Channel legend (same in every plot):
  Blue  = PPG1_SUB1  (IR  ~850 nm)   ← cardiac + tissue absorption
  Red   = PPG1_SUB2  (RED ~660 nm)   ← more sensitive to SpO2
  Green = PPG1_SUB3  (3rd channel)   ← likely green/ambient reference
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'output_quality')

AS7058_FILES = {
    'Wrist V1':  ('AS7058/01_Wrist_AS7058/V1/wrist_position_nikhil_02.032026.csv',              'Wrist',  'A'),
    'Wrist V2':  ('AS7058/01_Wrist_AS7058/v2/wrist_position_nikhil_V2_02.032026.csv',            'Wrist',  'A'),
    'Finger V1': ('AS7058/02_Finger_AS7058/V1/Finger_position_nikhil_V1_02.032026_2026-03-02_12-06-03.csv', 'Finger', 'B'),
    'Finger V2': ('AS7058/02_Finger_AS7058/V2/Finger_position_nikhil_V2_02.032026_2026-03-02_12-12-17.csv', 'Finger', 'B'),
    'Finger V3': ('AS7058/04_Finger_AS7058_Parallel with SP-20/Finger_position_nikhil_V3_02.032026_2026-03-02_14-09-26.csv', 'Finger', 'B'),
    'Chest V1':  ('AS7058/03_Chest_AS7058/V1_wrist algo/Chest_position_nikhil_V1_02.032026_2026-03-02_13-24-19.csv',  'Chest',  'A'),
    'Chest V2':  ('AS7058/03_Chest_AS7058/V2_wrist algo/Chest_position_nikhil_V2_02.032026_2026-03-02_13-36-02.csv',  'Chest',  'A'),
}
SP20_FILE = 'SP-20/SP-20 _20260302140253.csv'

SITE_COLOR = {'Wrist': '#1565C0', 'Finger': '#2E7D32', 'Chest': '#B71C1C'}
CH_COLOR   = {'IR': '#1565C0', 'RED': '#C62828', '3rd': '#2E7D32'}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {name}")


def bp(sig, fs, lo=0.5, hi=4.0, order=4):
    """Butterworth bandpass filter (zero-phase)."""
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order, [lo / nyq, min(hi / nyq, 0.999)], btype='band')
    return scipy_signal.filtfilt(b, a, sig)


def hp(sig, fs, cutoff=0.05, order=2):
    """Highpass to remove slow baseline drift (DC removal)."""
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order, max(cutoff / nyq, 0.001), btype='high')
    return scipy_signal.filtfilt(b, a, sig)


def snr_db(sig, fs):
    """SNR = HR-band power (0.7–3.5 Hz) / noise-band power (3.5–8 Hz)."""
    f, p = scipy_signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 8)))
    s = np.trapz(p[(f >= 0.7) & (f <= 3.5)], f[(f >= 0.7) & (f <= 3.5)])
    n = np.trapz(p[(f >= 3.5) & (f <= 8.0)], f[(f >= 3.5) & (f <= 8.0)])
    return float(10 * np.log10(s / n)) if n > 0 else np.nan


def dominant_hr(sig, fs):
    """Dominant frequency in HR band → HR in BPM."""
    f, p = scipy_signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 8)))
    mask = (f >= 0.7) & (f <= 3.5)
    if not np.any(mask):
        return np.nan
    return float(f[mask][np.argmax(p[mask])] * 60.0)


def ac_dc_ratio(sig, fs):
    """AC/DC ratio (in %) — perfusion index proxy from raw signal."""
    dc = np.mean(np.abs(sig))
    if dc == 0:
        return np.nan
    ac_sig = bp(sig, fs, 0.5, 4.0)
    ac = np.std(ac_sig)
    return float(ac / dc * 100.0)


def baseline_stability(sig):
    """Std of 5-second block means — lower = more stable baseline."""
    return float(np.std(sig))


def peak_detect(filt_sig, fs):
    """Returns (peak_indices, rr_ms array)."""
    dist = int(fs * 0.4)
    thr  = 0.3 * np.nanmax(filt_sig)
    peaks, _ = scipy_signal.find_peaks(filt_sig, distance=dist, height=thr)
    rr_ms = np.diff(peaks) / fs * 1000.0 if len(peaks) > 1 else np.array([])
    return peaks, rr_ms

# ─── DATA LOADER ──────────────────────────────────────────────────────────────

def load_dataset(rel_path, label):
    """
    Load one AS7058 CSV.  Returns dict:
      ppg_df   – DataFrame with columns: t, IR, RED, THIRD (PPG rows only)
      acc_df   – DataFrame with columns: t, ax, ay, az  (Schema A only)
      agc_df   – DataFrame with columns: t, agc1_led, agc2_led
      fs       – PPG sampling rate (Hz)
      schema   – 'A' or 'B'
      label    – str
    """
    fpath = os.path.join(BASE, rel_path)
    if not os.path.isfile(fpath):
        print(f"  [MISSING] {fpath}")
        return None

    print(f"  Loading {label} ...")
    raw = pd.read_csv(fpath, low_memory=False)

    TS = 'TIMESTAMP [s]'
    schema = 'A' if 'ACC_X' in raw.columns else 'B'

    # ── PPG rows ──────────────────────────────────────────────────────────────
    ppg_mask = raw['PPG1_SUB1'].notna() & (raw['PPG1_SUB1'].astype(str).str.strip() != '')
    ppg_raw  = raw[ppg_mask].copy()
    ppg_raw[TS]         = pd.to_numeric(ppg_raw[TS],          errors='coerce')
    ppg_raw['PPG1_SUB1'] = pd.to_numeric(ppg_raw['PPG1_SUB1'], errors='coerce')
    ppg_raw['PPG1_SUB2'] = pd.to_numeric(ppg_raw['PPG1_SUB2'], errors='coerce').fillna(0)
    ppg_raw['PPG1_SUB3'] = pd.to_numeric(ppg_raw['PPG1_SUB3'], errors='coerce').fillna(0)
    ppg_raw = ppg_raw.dropna(subset=[TS, 'PPG1_SUB1']).sort_values(TS).reset_index(drop=True)

    # Infer sampling rate
    diffs = ppg_raw[TS].diff().dropna()
    fs = round(1.0 / diffs.median()) if diffs.median() > 0 else 100.0

    # Elapsed time from zero
    t0  = ppg_raw[TS].iloc[0]
    t   = (ppg_raw[TS] - t0).values
    ir  = ppg_raw['PPG1_SUB1'].values.astype(float)
    red = ppg_raw['PPG1_SUB2'].values.astype(float)
    th3 = ppg_raw['PPG1_SUB3'].values.astype(float)

    # Mask out the AGC settling period (first 2s where RED == 0)
    valid_red = red != 0
    red = np.where(valid_red, red, np.nan)
    th3 = np.where(th3 != 0, th3, np.nan)

    ppg_df = pd.DataFrame({'t': t, 'IR': ir, 'RED': red, 'THIRD': th3})

    # ── ACC rows (Schema A) ───────────────────────────────────────────────────
    acc_df = pd.DataFrame()
    if schema == 'A' and 'ACC_X' in raw.columns:
        amask = raw['ACC_X'].notna() & (raw['ACC_X'].astype(str).str.strip() != '')
        a = raw[amask].copy()
        a[TS] = pd.to_numeric(a[TS], errors='coerce')
        for c in ['ACC_X', 'ACC_Y', 'ACC_Z']:
            a[c] = pd.to_numeric(a[c], errors='coerce')
        a = a.dropna(subset=[TS, 'ACC_X']).sort_values(TS)
        acc_df = pd.DataFrame({
            't':  (a[TS] - t0).values,
            'ax': a['ACC_X'].values.astype(float),
            'ay': a['ACC_Y'].values.astype(float),
            'az': a['ACC_Z'].values.astype(float),
        })

    # ── AGC rows ──────────────────────────────────────────────────────────────
    agc_df = pd.DataFrame()
    if 'AGC1_LED_CURRENT' in raw.columns:
        amask = raw['AGC1_LED_CURRENT'].notna() & (raw['AGC1_LED_CURRENT'].astype(str).str.strip() != '')
        a = raw[amask].copy()
        a[TS] = pd.to_numeric(a[TS], errors='coerce')
        a['AGC1_LED_CURRENT'] = pd.to_numeric(a['AGC1_LED_CURRENT'], errors='coerce')
        a['AGC2_LED_CURRENT'] = pd.to_numeric(a.get('AGC2_LED_CURRENT', pd.Series(dtype=float)),
                                               errors='coerce')
        a = a.dropna(subset=[TS, 'AGC1_LED_CURRENT']).sort_values(TS)
        agc_df = pd.DataFrame({
            't':    (a[TS] - t0).values,
            'agc1': a['AGC1_LED_CURRENT'].values.astype(float),
            'agc2': a['AGC2_LED_CURRENT'].values.astype(float),
        })

    return {
        'ppg_df': ppg_df, 'acc_df': acc_df, 'agc_df': agc_df,
        'fs': fs, 'schema': schema, 'label': label,
        'duration': t[-1] if len(t) else 0,
        'site': label.split()[0],
    }


def load_sp20(rel_path):
    fpath = os.path.join(BASE, rel_path)
    if not os.path.isfile(fpath):
        return None
    print("  Loading SP-20 ...")
    df = pd.read_csv(fpath)
    df.columns = [c.strip() for c in df.columns]
    df['ts'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S %b %d %Y', errors='coerce')
    df = df.dropna(subset=['ts']).reset_index(drop=True)
    df['t']    = (df['ts'] - df['ts'].iloc[0]).dt.total_seconds()
    df['spo2'] = pd.to_numeric(df['Oxygen Level'], errors='coerce')
    df['hr']   = pd.to_numeric(df['Pulse Rate'],   errors='coerce')
    # Drop "--" rows
    df = df[df['spo2'].notna() & df['hr'].notna()].reset_index(drop=True)
    return df[['t', 'spo2', 'hr']]


# ─── QUALITY SCORECARD ────────────────────────────────────────────────────────

def compute_quality(ds):
    """Compute all quality metrics for one dataset. Returns dict."""
    ppg = ds['ppg_df']
    fs  = ds['fs']
    t   = ppg['t'].values
    ir  = ppg['IR'].values

    # Use only data after AGC settling (skip first 2 s)
    ok = t >= 2.0
    ir_stable = ir[ok]

    # IR channel metrics
    dc_ir = np.mean(ir_stable)
    ir_hp = hp(ir_stable, fs, cutoff=0.05) if len(ir_stable) > int(fs * 4) else ir_stable
    ir_bp = bp(ir_stable, fs, 0.5, 4.0)   if len(ir_stable) > int(fs * 4) else ir_stable

    q = {
        'Duration (s)':       round(ds['duration'], 1),
        'Fs (Hz)':            int(fs),
        'Schema':             ds['schema'],

        # IR channel
        'IR DC (ADC)':        int(dc_ir),
        'IR AC peak-peak (ADC)': int(np.ptp(ir_bp)) if len(ir_bp) > 0 else 0,
        'IR AC/DC (%)':       round(ac_dc_ratio(ir_stable, fs), 4) if len(ir_stable) > int(fs*4) else 0,
        'IR SNR (dB)':        round(snr_db(ir_hp, fs), 2) if len(ir_hp) > int(fs*4) else np.nan,
        'Dominant HR (BPM)':  round(dominant_hr(ir_bp, fs), 1) if len(ir_bp) > int(fs*4) else np.nan,

        # RED channel
        'RED DC (ADC)':       0,
        'RED AC/DC (%)':      0,

        # AGC
        'AGC1 LED mA (final)': np.nan,
        'AGC2 LED mA (final)': np.nan,

        # Stability
        'IR baseline std':    round(np.std(ir_stable / dc_ir * 100), 4) if dc_ir > 0 else np.nan,
    }

    # RED metrics (skip NaN/zero)
    red = ppg['RED'].values
    red_ok = red[ok & ~np.isnan(red) & (red > 0)]
    if len(red_ok) > int(fs * 4):
        q['RED DC (ADC)']   = int(np.mean(red_ok))
        q['RED AC/DC (%)']  = round(ac_dc_ratio(red_ok, fs), 4)

    # AGC final current
    agc = ds['agc_df']
    if len(agc) > 0:
        q['AGC1 LED mA (final)'] = int(agc['agc1'].iloc[-1])
        q['AGC2 LED mA (final)'] = int(agc['agc2'].iloc[-1]) if 'agc2' in agc.columns else np.nan

    return q


# ─── PLOT A: Per-dataset detailed channel plots ───────────────────────────────

def plot_dataset_detail(ds, quality):
    """
    3-column layout for one dataset:
      Col 0: Raw ADC counts of IR + RED + 3rd channel (first 60 s)
      Col 1: AC-filtered (highpass 0.05 Hz) IR and RED showing heartbeat waveform
      Col 2: PSD of IR AC signal (0–5 Hz)  showing HR peak
    """
    label = ds['label']
    site  = ds['site']
    ppg   = ds['ppg_df']
    fs    = ds['fs']
    t     = ppg['t'].values
    ir    = ppg['IR'].values.astype(float)
    red   = ppg['RED'].values.astype(float)
    th3   = ppg['THIRD'].values.astype(float)

    # Use first 60 s for display
    win = t <= 60.0
    t_d   = t[win]
    ir_d  = ir[win]
    red_d = red[win]
    th3_d = th3[win]

    # AC-filtered (remove baseline drift; keep heartbeat band)
    ir_ac  = np.full_like(ir_d,  np.nan)
    red_ac = np.full_like(red_d, np.nan)
    if np.sum(~np.isnan(ir_d)) > int(fs * 4):
        ir_ac = hp(np.nan_to_num(ir_d, nan=np.nanmean(ir_d)), fs, 0.05)
    red_valid = ~np.isnan(red_d) & (red_d > 0)
    if np.sum(red_valid) > int(fs * 4):
        r_fill = red_d.copy()
        r_fill[~red_valid] = np.nanmean(red_d[red_valid])
        red_ac = hp(r_fill, fs, 0.05)
        red_ac[~red_valid] = np.nan

    # PSD on full AC signal (after 2 s settling)
    ok = t >= 2.0
    ir_full_ac = np.full(np.sum(ok), np.nan)
    if np.sum(ok) > int(fs * 4):
        ir_full_ac = hp(np.nan_to_num(ir[ok], nan=np.nanmean(ir[ok])), fs, 0.05)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(
        f'{label}  |  Site: {site}  |  Schema {ds["schema"]}  |  '
        f'Fs={int(fs)} Hz  |  Duration={ds["duration"]:.0f}s  |  '
        f'IR SNR={quality["IR SNR (dB)"]:+.1f} dB  |  '
        f'AGC1={quality["AGC1 LED mA (final)"]} LSB',
        fontsize=10, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    c = SITE_COLOR[site]

    # ── Col 0: Raw ADC (0–60 s) ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])

    ax0.plot(t_d, ir_d,  color=CH_COLOR['IR'],  lw=0.5, label='IR  (SUB1 ~850 nm)', alpha=0.9)
    if np.any(red_valid[win if len(red_valid) == len(win) else slice(None)]):
        ax0.plot(t_d, red_d, color=CH_COLOR['RED'], lw=0.5, label='RED (SUB2 ~660 nm)', alpha=0.8)
    th3_valid = ~np.isnan(th3_d) & (th3_d > 0)
    if np.any(th3_valid):
        ax0.plot(t_d, th3_d, color=CH_COLOR['3rd'], lw=0.5, label='3rd (SUB3)',         alpha=0.7)

    ax0.set_title('Raw ADC Signal (0 – 60 s)', fontsize=9)
    ax0.set_xlabel('Time (seconds)', fontsize=8)
    ax0.set_ylabel('ADC Counts (LSB)', fontsize=8)
    ax0.legend(fontsize=7, loc='upper right')
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(labelsize=7)
    ax0.annotate('← DC baseline (large offset)\nSmall AC ripple = heartbeat',
                 xy=(0.02, 0.05), xycoords='axes fraction', fontsize=6.5, color='gray')

    # ── Col 1: AC-filtered (heartbeat waveform) ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])

    ax1.plot(t_d, ir_ac, color=CH_COLOR['IR'],  lw=0.7, label='IR  AC',  alpha=0.9)
    ax1.plot(t_d, red_ac, color=CH_COLOR['RED'], lw=0.7, label='RED AC', alpha=0.8)
    ax1.axhline(0, color='gray', lw=0.5, linestyle='--')

    # Mark detected peaks on IR
    try:
        ir_bp_d = bp(np.nan_to_num(ir_ac, 0), fs, 0.5, 4.0)
        pk, _ = peak_detect(ir_bp_d, fs)
        pk_in_window = pk[t_d[pk] <= 60.0] if len(pk) else np.array([], dtype=int)
        if len(pk_in_window):
            ax1.plot(t_d[pk_in_window], ir_ac[pk_in_window], 'v',
                     color='orange', ms=4, zorder=5, label=f'{len(pk_in_window)} peaks')
    except Exception:
        pass

    ax1.set_title('AC-Filtered Signal (Heartbeat Waveform)', fontsize=9)
    ax1.set_xlabel('Time (seconds)', fontsize=8)
    ax1.set_ylabel('ΔADC Counts (AC component)', fontsize=8)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)

    # ── Col 2: PSD ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])

    if np.sum(ok) > int(fs * 4) and not np.all(np.isnan(ir_full_ac)):
        nperseg = min(len(ir_full_ac), int(fs * 8))
        f, p = scipy_signal.welch(ir_full_ac, fs=fs, nperseg=nperseg)
        ax2.semilogy(f, p, color=CH_COLOR['IR'], lw=1.2, label='IR PSD')

        # HR peak
        mask = (f >= 0.7) & (f <= 3.5)
        if np.any(mask):
            dom_f = f[mask][np.argmax(p[mask])]
            ax2.axvline(dom_f, color='orange', lw=1.2, linestyle='--',
                        label=f'HR≈{dom_f*60:.0f} BPM')

        # Band shading
        ax2.axvspan(0.1,  0.5,  alpha=0.08, color='purple', label='RR band (6–30 br/min)')
        ax2.axvspan(0.7,  3.5,  alpha=0.08, color='green',  label='HR band (42–210 BPM)')

    ax2.set_xlim(0, 5)
    ax2.set_title('Power Spectral Density of IR (AC)', fontsize=9)
    ax2.set_xlabel('Frequency (Hz)', fontsize=8)
    ax2.set_ylabel('PSD (ADC² / Hz)', fontsize=8)
    ax2.legend(fontsize=6.5)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)

    return fig


# ─── PLOT B: Quality scorecard table ─────────────────────────────────────────

def plot_quality_scorecard(all_quality):
    """Horizontal bar + table combining all dataset metrics."""
    labels  = list(all_quality.keys())
    metrics = ['IR SNR (dB)', 'IR AC/DC (%)', 'RED AC/DC (%)',
               'Dominant HR (BPM)', 'AGC1 LED mA (final)', 'Duration (s)']

    # Color by site
    sites = [lbl.split()[0] for lbl in labels]
    colors = [SITE_COLOR[s] for s in sites]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Data Quality Scorecard — All Datasets', fontsize=13, fontweight='bold')

    thresholds = {
        'IR SNR (dB)':          6.0,
        'IR AC/DC (%)':         0.05,
        'RED AC/DC (%)':        0.05,
        'Dominant HR (BPM)':    None,
        'AGC1 LED mA (final)':  None,
        'Duration (s)':         300.0,
    }
    ylabels = {
        'IR SNR (dB)':          'IR SNR (dB)\n[>6 dB = usable]',
        'IR AC/DC (%)':         'IR AC/DC Ratio (%)\n[Perfusion Index proxy]',
        'RED AC/DC (%)':        'RED AC/DC Ratio (%)\n[SpO2 algorithm input]',
        'Dominant HR (BPM)':    'Estimated HR (BPM)\n[from PSD peak]',
        'AGC1 LED mA (final)': 'AGC1 LED Current (LSB)\n[lower = better coupling]',
        'Duration (s)':         'Recording Duration (s)\n[>300 s = sufficient]',
    }

    for ax, metric in zip(axes.flat, metrics):
        vals = [all_quality[lbl].get(metric, np.nan) for lbl in labels]
        vals_plot = [v if np.isfinite(v) else 0 for v in vals]

        bars = ax.barh(labels, vals_plot, color=colors, alpha=0.75, edgecolor='black', linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}', va='center', ha='left', fontsize=7.5)

        # Threshold line
        thr = thresholds[metric]
        if thr is not None:
            ax.axvline(thr, color='red', lw=1.2, linestyle='--', label=f'Threshold = {thr}')
            ax.legend(fontsize=7)

        ax.set_title(ylabels[metric], fontsize=8.5, fontweight='bold')
        ax.set_xlabel(metric.split('(')[0].strip(), fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(labelsize=7)

        # Color-code quality
        if metric == 'IR SNR (dB)':
            for bar, val in zip(bars, vals):
                if np.isfinite(val) and val < 6.0:
                    bar.set_hatch('///')
                    bar.set_alpha(0.4)

    # Legend for sites
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SITE_COLOR[s], label=s) for s in ['Wrist', 'Finger', 'Chest']]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
               title='Body Location', title_fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    return fig


# ─── PLOT C: Cross-location AC waveform comparison ───────────────────────────

def plot_cross_location(datasets):
    """
    Overlay IR AC-filtered waveforms from the best V of each site.
    Shows at a glance how heartbeat amplitude and shape differ per body location.
    """
    reps = {'Wrist': 'Wrist V1', 'Finger': 'Finger V1', 'Chest': 'Chest V1'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        'IR AC-Filtered Heartbeat Waveform Comparison by Body Location\n'
        'X = Time (seconds) | Y = ΔADCounts (pulsatile AC component, baseline removed)',
        fontsize=11, fontweight='bold'
    )

    for ax, (site, lbl) in zip(axes, reps.items()):
        ds = datasets.get(lbl)
        if ds is None:
            ax.set_visible(False)
            continue

        ppg = ds['ppg_df']
        fs  = ds['fs']
        t   = ppg['t'].values
        ir  = ppg['IR'].values.astype(float)

        # Skip first 2 s (AGC settling)
        ok = t >= 2.0
        t_ok  = t[ok] - t[ok][0]
        ir_ok = ir[ok]

        # AC filter
        if len(ir_ok) > int(fs * 4):
            ir_ac = hp(np.nan_to_num(ir_ok, nan=np.nanmean(ir_ok)), fs, 0.05)
        else:
            ir_ac = ir_ok - np.mean(ir_ok)

        c = SITE_COLOR[site]
        ax.plot(t_ok, ir_ac, color=c, lw=0.6, alpha=0.85, label=lbl)

        # Peaks
        try:
            ir_bp_f = bp(ir_ac, fs, 0.5, 4.0)
            pk, rr = peak_detect(ir_bp_f, fs)
            if len(pk):
                ax.plot(t_ok[pk], ir_ac[pk], 'v', color='orange', ms=3, zorder=5,
                        label=f'{len(pk)} peaks | med RR={np.median(rr):.0f} ms = {60000/np.median(rr):.0f} BPM' if len(rr) else f'{len(pk)} peaks')
        except Exception:
            pass

        # Amplitude annotation
        amp = np.ptp(ir_ac) if len(ir_ac) else 0
        ax.set_title(
            f'{site}  ({lbl})  |  Peak-to-Peak AC = {amp:.0f} ADC counts  |  '
            f'Fs = {int(fs)} Hz  |  Duration = {ds["duration"]:.0f} s',
            fontsize=9, color=c, fontweight='bold'
        )
        ax.set_ylabel('ΔADC Counts\n(AC, baseline removed)', fontsize=8)
        ax.axhline(0, color='gray', lw=0.5, linestyle='--')
        ax.legend(fontsize=7.5, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel('Time (seconds)', fontsize=9)
    fig.tight_layout()
    return fig


# ─── PLOT D: SP-20 trend ──────────────────────────────────────────────────────

def plot_sp20_detail(sp20):
    """
    Two-panel SP-20 plot:
      Top:    Heart Rate (BPM) over time
      Bottom: SpO2 (%) over time
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(
        'SP-20 Reference Device — Full 6-Minute Recording\n'
        'X = Time (seconds from start) | Y = measured value at 1 Hz',
        fontsize=11, fontweight='bold'
    )

    t    = sp20['t'].values
    hr   = sp20['hr'].values
    spo2 = sp20['spo2'].values

    # HR
    ax1.plot(t, hr, color='#D84315', lw=1.5, marker='o', ms=2.5, label='Pulse Rate')
    ax1.axhline(np.nanmean(hr), color='gray', lw=1.0, linestyle='--',
                label=f'Mean = {np.nanmean(hr):.1f} BPM')
    ax1.fill_between(t, np.nanmean(hr) - np.nanstd(hr),
                         np.nanmean(hr) + np.nanstd(hr),
                     color='#D84315', alpha=0.12, label=f'±1σ = {np.nanstd(hr):.1f} BPM')
    ax1.set_ylabel('Heart Rate (BPM)', fontsize=9)
    ax1.set_ylim(max(0, np.nanmin(hr) - 10), np.nanmax(hr) + 10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=8)
    ax1.set_title(f'HR: mean={np.nanmean(hr):.1f} BPM, std={np.nanstd(hr):.1f}, '
                  f'min={np.nanmin(hr):.0f}, max={np.nanmax(hr):.0f}', fontsize=9)

    # SpO2
    ax2.plot(t, spo2, color='#1565C0', lw=1.5, marker='o', ms=2.5, label='SpO2')
    ax2.axhline(np.nanmean(spo2), color='gray', lw=1.0, linestyle='--',
                label=f'Mean = {np.nanmean(spo2):.2f}%')
    ax2.axhline(95, color='red', lw=1.0, linestyle=':', label='95% clinical threshold')
    ax2.fill_between(t, np.nanmean(spo2) - np.nanstd(spo2),
                         np.nanmean(spo2) + np.nanstd(spo2),
                     color='#1565C0', alpha=0.12, label=f'±1σ = {np.nanstd(spo2):.2f}%')
    ax2.set_ylabel('SpO2 (%)', fontsize=9)
    ax2.set_ylim(90, 102)
    ax2.set_xlabel('Time (seconds from start)', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)
    ax2.set_title(f'SpO2: mean={np.nanmean(spo2):.2f}%, std={np.nanstd(spo2):.2f}%, '
                  f'min={np.nanmin(spo2):.0f}%, max={np.nanmax(spo2):.0f}%', fontsize=9)

    fig.tight_layout()
    return fig


# ─── PLOT E: AGC current vs site ──────────────────────────────────────────────

def plot_agc_comparison(datasets):
    """
    AGC LED current over time for each dataset.
    Lower final current = sensor found strong signal = better optical contact.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    fig.suptitle(
        'AGC LED Current Over Time by Body Location\n'
        'X = Time (seconds) | Y = LED Drive Current (LSB units, 1 LSB ≈ 0.4 mA for AS7058)\n'
        'Lower settled current → better tissue-sensor optical coupling',
        fontsize=10, fontweight='bold'
    )

    site_reps = [
        ('Wrist',  ['Wrist V1',  'Wrist V2']),
        ('Finger', ['Finger V1', 'Finger V2', 'Finger V3']),
        ('Chest',  ['Chest V1',  'Chest V2']),
    ]

    for ax, (site, lbls) in zip(axes, site_reps):
        plotted = False
        for lbl in lbls:
            ds = datasets.get(lbl)
            if ds is None:
                continue
            agc = ds['agc_df']
            if len(agc) == 0:
                continue
            color = SITE_COLOR[site]
            ls = '-' if 'V1' in lbl else '--'
            ax.plot(agc['t'], agc['agc1'], color=color, lw=1.4, linestyle=ls,
                    label=f'{lbl} AGC1 (IR)')
            if 'agc2' in agc.columns and not agc['agc2'].isna().all():
                ax.plot(agc['t'], agc['agc2'], color=color, lw=1.0, linestyle=ls,
                        alpha=0.5, label=f'{lbl} AGC2')
            plotted = True

        ax.set_title(f'{site} — AGC LED Current', fontsize=9, fontweight='bold',
                     color=SITE_COLOR[site])
        ax.set_ylabel('LED Current (LSB)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, ncol=3)
        ax.tick_params(labelsize=7)
        if not plotted:
            ax.text(0.5, 0.5, 'No AGC data (Schema B files)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')

    axes[-1].set_xlabel('Time (seconds)', fontsize=9)
    fig.tight_layout()
    return fig


# ─── PLOT F: Per-dataset 10-second zoom into heartbeat ────────────────────────

def plot_heartbeat_zoom(datasets):
    """
    Zoom into 10 seconds of clean signal for each dataset.
    Shows clearly whether individual heartbeats are resolved.
    """
    labels = list(datasets.keys())
    n = len(labels)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle(
        'Heartbeat Waveform — 10-Second Zoom (Seconds 20–30)\n'
        'X = Time (seconds) | Y = ΔADC (AC-filtered IR, highpass 0.05 Hz)\n'
        'Each "hump" = one heartbeat. Clear humps = good signal quality.',
        fontsize=10, fontweight='bold'
    )
    axes = axes.flat

    for ax, lbl in zip(axes, labels):
        ds = datasets[lbl]
        ppg = ds['ppg_df']
        fs  = ds['fs']
        t   = ppg['t'].values
        ir  = ppg['IR'].values.astype(float)
        site = ds['site']
        color = SITE_COLOR[site]

        # Zoom: seconds 20–30 (after AGC settled)
        win = (t >= 20.0) & (t <= 30.0)
        if np.sum(win) < int(fs * 2):
            win = (t >= 2.0) & (t <= 12.0)

        t_z  = t[win] - t[win][0]
        ir_z = ir[win]
        if len(ir_z) > int(fs * 2):
            ir_ac = hp(np.nan_to_num(ir_z, nan=np.nanmean(ir_z)), fs, 0.05)
        else:
            ir_ac = ir_z - np.mean(ir_z)

        ax.plot(t_z, ir_ac, color=color, lw=0.9, alpha=0.9)
        ax.axhline(0, color='gray', lw=0.5, linestyle='--')

        # Peaks in this window
        try:
            ir_bp_z = bp(ir_ac, fs, 0.5, 4.0)
            pk, rr = peak_detect(ir_bp_z, fs)
            if len(pk):
                ax.plot(t_z[pk], ir_ac[pk], 'v', color='orange', ms=5, zorder=5)
                hr_est = 60000.0 / np.median(rr) if len(rr) else np.nan
                hr_str = f'{hr_est:.0f} BPM' if np.isfinite(hr_est) else 'N/A'
            else:
                hr_str = 'No peaks'
        except Exception:
            hr_str = 'Error'

        amp = np.ptp(ir_ac)
        ax.set_title(f'{lbl}  |  Amp={amp:.0f} ADC  |  Est. HR={hr_str}', fontsize=8.5)
        ax.set_xlabel('Time (s) — window: t+20 to t+30', fontsize=7.5)
        ax.set_ylabel('ΔADC Counts', fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # Quality badge
        snr = snr_db(ir_ac, fs) if len(ir_ac) > int(fs * 2) else np.nan
        badge_color = 'limegreen' if snr >= 6 else ('orange' if snr >= 3 else 'red')
        badge_text  = f'SNR={snr:+.1f}dB' if np.isfinite(snr) else 'SNR=N/A'
        ax.text(0.98, 0.95, badge_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.85))

    # Hide unused axes
    for ax in list(axes)[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 65)
    print(' PPG Quality Analysis - AS7058 + SP-20')
    print('=' * 65)

    # ── Load all datasets ──────────────────────────────────────────────────────
    print('\n[1] Loading datasets ...')
    datasets = {}
    for label, (relpath, site, schema) in AS7058_FILES.items():
        ds = load_dataset(relpath, label)
        if ds is not None:
            datasets[label] = ds

    sp20 = load_sp20(SP20_FILE)

    # ── Compute quality metrics ────────────────────────────────────────────────
    print('\n[2] Computing quality metrics ...')
    all_quality = {}
    for lbl, ds in datasets.items():
        all_quality[lbl] = compute_quality(ds)

    # ── Print quality table ────────────────────────────────────────────────────
    print('\n' + '-' * 100)
    print(f"{'Dataset':<14} {'Site':<8} {'Dur(s)':<8} {'Fs':<6} {'IR DC':<10} {'IR SNR(dB)':<12} "
          f"{'IR AC/DC%':<12} {'RED DC':<10} {'RED AC/DC%':<12} {'Est.HR':<9} {'AGC1 LSB'}")
    print('-' * 100)
    for lbl, q in all_quality.items():
        site = lbl.split()[0]
        print(f"{lbl:<14} {site:<8} {q['Duration (s)']:<8.0f} {q['Fs (Hz)']:<6} "
              f"{q['IR DC (ADC)']:<10} {q['IR SNR (dB)']:<12.2f} {q['IR AC/DC (%)']:<12.4f} "
              f"{q['RED DC (ADC)']:<10} {q['RED AC/DC (%)']:<12.4f} "
              f"{q['Dominant HR (BPM)']:<9.1f} {q['AGC1 LED mA (final)']}")
    print('-' * 100)

    # ── Generate plots ─────────────────────────────────────────────────────────
    print('\n[3] Generating plots ...')

    # A: Per-dataset detail
    for lbl, ds in datasets.items():
        fig = plot_dataset_detail(ds, all_quality[lbl])
        fname = f"A_{lbl.replace(' ', '_')}_detail.png"
        _save(fig, fname)

    # B: Scorecard
    fig = plot_quality_scorecard(all_quality)
    _save(fig, 'B_quality_scorecard.png')

    # C: Cross-location comparison
    fig = plot_cross_location(datasets)
    _save(fig, 'C_cross_location_comparison.png')

    # D: SP-20 detail
    if sp20 is not None:
        fig = plot_sp20_detail(sp20)
        _save(fig, 'D_SP20_trend.png')

    # E: AGC comparison
    fig = plot_agc_comparison(datasets)
    _save(fig, 'E_agc_comparison.png')

    # F: Heartbeat zoom
    fig = plot_heartbeat_zoom(datasets)
    _save(fig, 'F_heartbeat_zoom.png')

    print(f'\n All plots saved to: {OUT}')
    print(' Done.\n')


if __name__ == '__main__':
    main()
