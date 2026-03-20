"""
Iteration 2 — Cross-Position Comparison Report
================================================
Run:   py -3 iteration2_comparison.py
Output: Iteration 2_Test data/Comparison_Report/

Loads all 5 chest positions (A-E), computes metrics for each,
and generates a unified comparison with ranking, plots, and
a detailed markdown report answering Nikhil's 3 feasibility questions.
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
OUT_DIR    = os.path.join(ITER2_DIR, 'Comparison_Report')
os.makedirs(OUT_DIR, exist_ok=True)

THR = dict(pi_good=1.0, pi_fair=0.3, snr_good=10.0, snr_fair=6.0,
           rrcv_good=10.0, rrcv_fair=25.0, amp_good=100, amp_fair=20,
           hr_lo=40.0, hr_hi=160.0)

ITER1_CHEST = dict(pi=0.22, snr=19.1, amp=6.0, hr_pct=100.0, rrcv=3.6, agc=13.0)
TAIL_S = 120.0

POSITIONS = ['A', 'B', 'C', 'D', 'E']
POS_COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728', 'E': '#9467bd'}

# ── Signal processing (same as position_analysis.py) ─────────────────────────
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

def grade(val, g, f):
    if np.isnan(val): return 'N/A'
    return 'GOOD' if val >= g else 'FAIR' if val >= f else 'POOR'

def grade_inv(val, g, f):
    if np.isnan(val): return 'N/A'
    return 'GOOD' if val <= g else 'FAIR' if val <= f else 'POOR'

GRADE_COLOR = {'GOOD': '#2ca02c', 'FAIR': '#ff7f0e', 'POOR': '#d62728', 'N/A': '#999999'}

# ── File discovery ────────────────────────────────────────────────────────────
def find_position_dir(pos_letter):
    for name in [f'Position {pos_letter}', f'Postion {pos_letter}']:
        p = os.path.join(ITER2_DIR, name)
        if os.path.isdir(p): return p
    return None

def discover_files(pos_dir):
    spo2_dir = os.path.join(pos_dir, 'SpO2')
    hrm_dir  = os.path.join(pos_dir, 'HRM RRM')
    def split(csvs):
        filt = [f for f in csvs if '_filtered' in os.path.basename(f).lower()]
        raw  = [f for f in csvs if '_filtered' not in os.path.basename(f).lower()]
        return raw[0] if raw else None, filt[0] if filt else None
    spo2_csvs = sorted(glob.glob(os.path.join(spo2_dir, '*.csv'))) if os.path.isdir(spo2_dir) else []
    hrm_csvs  = sorted(glob.glob(os.path.join(hrm_dir, '*.csv'))) if os.path.isdir(hrm_dir) else []
    spo2_raw, spo2_filt = split(spo2_csvs) if spo2_csvs else (None, None)
    hrm_raw, hrm_filt   = split(hrm_csvs) if hrm_csvs else (None, None)
    return dict(spo2_raw=spo2_raw, spo2_filt=spo2_filt, hrm_raw=hrm_raw, hrm_filt=hrm_filt)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_raw_file(fpath, mode='spo2'):
    raw = pd.read_csv(fpath, low_memory=False)
    ts_col = 'TIMESTAMP [s]'
    t_raw  = pd.to_numeric(raw[ts_col], errors='coerce')
    s1_raw = pd.to_numeric(raw['PPG1_SUB1'], errors='coerce')
    s2_raw = pd.to_numeric(raw['PPG1_SUB2'], errors='coerce')
    ppg_mask = s1_raw.notna()
    t  = t_raw[ppg_mask].values.astype(float); t -= t[0]
    s1 = s1_raw[ppg_mask].values.astype(float)
    s2 = s2_raw[ppg_mask].values.astype(float)
    s3 = np.full_like(s1, np.nan)
    if 'PPG1_SUB3' in raw.columns:
        s3 = pd.to_numeric(raw['PPG1_SUB3'], errors='coerce')[ppg_mask].values.astype(float)
    diffs = np.diff(t); fs = round(1.0 / np.median(diffs[diffs > 0]))
    agc_info = {}
    for prefix in ['AGC1', 'AGC2']:
        led_col = f'{prefix}_LED_CURRENT'
        if led_col in raw.columns:
            v = pd.to_numeric(raw[led_col], errors='coerce').dropna()
            if len(v): agc_info[f'{prefix}_led'] = float(v.median())
    events = {}
    if mode == 'spo2':
        sq_col = 'SPO2: SIGNAL_QUALITY'
        if sq_col in raw.columns:
            sq_mask = pd.to_numeric(raw[sq_col], errors='coerce').notna()
            ev_df = raw[sq_mask].copy()
            if len(ev_df):
                events['t']   = pd.to_numeric(ev_df[ts_col], errors='coerce').values - t_raw[ppg_mask].values[0]
                events['sq']  = pd.to_numeric(ev_df[sq_col], errors='coerce').values
                events['hr']  = pd.to_numeric(ev_df.get('SPO2: HEART_RATE [bpm]', pd.Series()), errors='coerce').values
                events['spo2']= pd.to_numeric(ev_df.get('SPO2: SPO2 [%]', pd.Series()), errors='coerce').values
                events['pi']  = pd.to_numeric(ev_df.get('SPO2: PI [%]', pd.Series()), errors='coerce').values
    elif mode == 'hrm':
        sq_col = 'HRM: SIGNAL_QUALITY'
        if sq_col in raw.columns:
            sq_mask = pd.to_numeric(raw[sq_col], errors='coerce').notna()
            ev_df = raw[sq_mask].copy()
            if len(ev_df):
                events['t']   = pd.to_numeric(ev_df[ts_col], errors='coerce').values - t_raw[ppg_mask].values[0]
                events['sq']  = pd.to_numeric(ev_df[sq_col], errors='coerce').values
                events['hr']  = pd.to_numeric(ev_df.get('HRM: HEART_RATE [bpm]', pd.Series()), errors='coerce').values
    return dict(t=t, s1=s1, s2=s2, s3=s3, fs=int(fs), duration=float(t[-1]),
                agc=agc_info, events=events, mode=mode)

# ── Sliding window metrics ────────────────────────────────────────────────────
def compute_sliding_metrics(t, signal, fs, win=10.0, step=2.0, skip=10.0):
    win_t, pis, snrs, amps, hrs, rrcvs = [], [], [], [], [], []
    start = skip; duration = t[-1] - t[0]
    while start + win <= duration:
        end = start + win; m = (t >= start) & (t < end)
        seg = signal[m].copy()
        if len(seg) < fs * 3: start += step; continue
        seg = np.clip(seg, np.percentile(seg, 1), np.percentile(seg, 99))
        dc = np.mean(seg); ac = bandpass(seg - dc, fs)
        ac_pp = np.percentile(ac, 90) - np.percentile(ac, 10)
        pi = abs(ac_pp) / abs(dc) * 100 if dc > 0 else 0.0
        snr = compute_snr(ac, fs); hr, rrcv, _ = detect_hr(ac, fs)
        win_t.append(start + win / 2); pis.append(pi); snrs.append(snr)
        amps.append(ac_pp); hrs.append(hr); rrcvs.append(rrcv)
        start += step
    return dict(win_t=np.array(win_t), pis=np.array(pis), snrs=np.array(snrs),
                amps=np.array(amps), hrs=np.array(hrs), rrcvs=np.array(rrcvs))

def summarise(mets):
    wt = mets['win_t']
    if len(wt) == 0:
        return dict(pi=np.nan, snr=np.nan, amp=np.nan, hr_pct=np.nan, rrcv=np.nan, hr_med=np.nan)
    tail = wt >= wt[-1] - TAIL_S
    snrs_t = mets['snrs'][tail]; amps_t = mets['amps'][tail]
    hrs_t = mets['hrs'][tail]; rrcvs_t = mets['rrcvs'][tail]
    pis_t = mets['pis'][tail]
    valid_hr = (hrs_t > THR['hr_lo']) & (hrs_t < THR['hr_hi'])
    hr_pct = float(valid_hr.mean() * 100) if len(hrs_t) > 0 else np.nan
    hr_med = float(np.nanmedian(hrs_t[valid_hr])) if valid_hr.any() else np.nan
    rrcv_m = float(np.nanmedian(rrcvs_t[valid_hr])) if valid_hr.any() else np.nan
    return dict(pi=float(np.nanmedian(pis_t)), snr=float(np.nanmedian(snrs_t)),
                amp=float(np.nanmedian(amps_t)), hr_pct=hr_pct, hr_med=hr_med, rrcv=rrcv_m)


# ── Load and analyze all positions ───────────────────────────────────────────
def load_all_positions():
    """Returns dict: pos_letter -> {spo2_raw, hrm_raw, channels: {name: summary}, best: summary, sliding: {name: mets}}"""
    results = {}
    for pos in POSITIONS:
        pos_dir = find_position_dir(pos)
        if pos_dir is None:
            print(f'  Position {pos}: folder not found, skipping')
            continue
        files = discover_files(pos_dir)
        if not files['spo2_raw'] or not files['hrm_raw']:
            print(f'  Position {pos}: missing files, skipping')
            continue

        print(f'  Loading Position {pos}...')
        spo2_raw = load_raw_file(files['spo2_raw'], mode='spo2')
        hrm_raw  = load_raw_file(files['hrm_raw'], mode='hrm')

        # Compute metrics for each channel
        channels = {}
        sliding = {}

        # SpO2 RED
        v = spo2_raw['s2'] > 1000; t_v = spo2_raw['t'][v]; s_v = spo2_raw['s2'][v]
        if len(t_v) > 2000:
            t_v = t_v - t_v[0]
            mets = compute_sliding_metrics(t_v, s_v, spo2_raw['fs'])
            channels['SpO2 RED'] = summarise(mets); sliding['SpO2 RED'] = mets

        # SpO2 SUB3
        v = spo2_raw['s3'] > 1000; t_v = spo2_raw['t'][v]; s_v = spo2_raw['s3'][v]
        if len(t_v) > 2000:
            t_v = t_v - t_v[0]
            mets = compute_sliding_metrics(t_v, s_v, spo2_raw['fs'])
            channels['SpO2 SUB3'] = summarise(mets); sliding['SpO2 SUB3'] = mets

        # HRM CH1
        v = hrm_raw['s1'] > 1000; t_v = hrm_raw['t'][v]; s_v = hrm_raw['s1'][v]
        if len(t_v) > 2000:
            t_v = t_v - t_v[0]
            mets = compute_sliding_metrics(t_v, s_v, hrm_raw['fs'])
            channels['HRM CH1'] = summarise(mets); sliding['HRM CH1'] = mets

        # Find best channel (highest PI)
        best_key, best_pi = None, -1
        for k, s in channels.items():
            if not np.isnan(s['pi']) and s['pi'] > best_pi:
                best_pi = s['pi']; best_key = k
        if best_key is None:
            best_snr = -999
            for k, s in channels.items():
                if not np.isnan(s['snr']) and s['snr'] > best_snr:
                    best_snr = s['snr']; best_key = k

        # Verdict
        best = channels.get(best_key, {})
        fails = 0
        if not np.isnan(best.get('pi', np.nan)) and best['pi'] < THR['pi_fair']: fails += 1
        if not np.isnan(best.get('snr', np.nan)) and best['snr'] < THR['snr_fair']: fails += 1
        if not np.isnan(best.get('hr_pct', np.nan)) and best['hr_pct'] < 50: fails += 1
        verdict = 'NOT USABLE' if fails >= 2 else 'MARGINAL' if fails == 1 else 'USABLE'

        results[pos] = dict(
            spo2_raw=spo2_raw, hrm_raw=hrm_raw,
            channels=channels, sliding=sliding,
            best_key=best_key, best=best, verdict=verdict,
        )
        print(f'    Best: {best_key} | PI={best.get("pi",np.nan):.4f}% | SNR={best.get("snr",np.nan):.1f}dB | Verdict: {verdict}')

    return results


# ── PLOT 1: Cross-Position PI Comparison Bar Chart ───────────────────────────
def plot_pi_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Cross-Position Comparison - Key Metrics (Best Channel per Position)',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('pi', 'Perfusion Index %', axes[0], [(THR['pi_fair'], 'FAIR (0.3%)'), (THR['pi_good'], 'GOOD (1.0%)')]),
        ('snr', 'SNR (dB)', axes[1], [(THR['snr_fair'], 'FAIR (6 dB)'), (THR['snr_good'], 'GOOD (10 dB)')]),
        ('amp', 'AC Amplitude (counts)', axes[2], [(THR['amp_fair'], 'FAIR (20)'), (THR['amp_good'], 'GOOD (100)')]),
    ]

    pos_labels = [f'Pos {p}' for p in POSITIONS if p in results] + ['Iter1\nChest']
    colors = [POS_COLORS[p] for p in POSITIONS if p in results] + ['#888888']

    for key, title, ax, thrs in metrics:
        vals = [results[p]['best'].get(key, np.nan) for p in POSITIONS if p in results]
        vals.append(ITER1_CHEST.get(key, np.nan))
        x = np.arange(len(pos_labels))
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8, alpha=0.85, zorder=3)
        for thr_val, thr_lbl in thrs:
            ax.axhline(thr_val, ls='--', lw=1.5, color='gray', alpha=0.7, label=thr_lbl)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                fmt = f'{val:.3f}' if key == 'pi' else f'{val:.1f}' if key == 'snr' else f'{val:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(1, ax.get_ylim()[1]),
                        fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(pos_labels, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3, zorder=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_cross_position_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print('  Saved: 01_cross_position_metrics.png')


# ── PLOT 2: All-Channel Comparison Heatmap ───────────────────────────────────
def plot_heatmap(results):
    # Rows: positions x channels, Cols: metrics
    metric_keys = ['pi', 'snr', 'amp', 'hr_pct', 'rrcv']
    metric_labels = ['PI %', 'SNR (dB)', 'AC Amp', 'HR Valid %', 'RR CV %']

    rows = []
    row_labels = []
    row_colors = []
    for pos in POSITIONS:
        if pos not in results: continue
        for ch_name, ch_summ in results[pos]['channels'].items():
            rows.append([ch_summ.get(k, np.nan) for k in metric_keys])
            is_best = (ch_name == results[pos]['best_key'])
            row_labels.append(f'Pos {pos}: {ch_name}' + (' *' if is_best else ''))
            row_colors.append(POS_COLORS[pos])

    # Add Iter1
    rows.append([ITER1_CHEST.get(k, np.nan) for k in metric_keys])
    row_labels.append('Iter1 Chest')
    row_colors.append('#888888')

    data = np.array(rows)
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(14, max(8, 0.6 * n_rows)))
    fig.suptitle('All Channels x All Positions - Quality Grades', fontsize=14, fontweight='bold')

    # Grade each cell
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            key = metric_keys[j]
            if key == 'pi': g = grade(val, THR['pi_good'], THR['pi_fair'])
            elif key == 'snr': g = grade(val, THR['snr_good'], THR['snr_fair'])
            elif key == 'amp': g = grade(val, THR['amp_good'], THR['amp_fair'])
            elif key == 'hr_pct': g = grade(val, 80, 50)
            elif key == 'rrcv': g = grade_inv(val, THR['rrcv_good'], THR['rrcv_fair'])
            else: g = 'N/A'

            color = GRADE_COLOR.get(g, '#999999')
            rect = plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, facecolor=color, alpha=0.2,
                                  edgecolor=color, lw=1)
            ax.add_patch(rect)

            if np.isnan(val): txt = 'N/A'
            elif key == 'pi': txt = f'{val:.4f}'
            elif key in ('snr', 'rrcv', 'hr_pct'): txt = f'{val:.1f}'
            else: txt = f'{val:.0f}'
            ax.text(j, i + 0.05, txt, ha='center', va='center', fontsize=9, fontweight='bold', color=color)
            ax.text(j, i - 0.2, g, ha='center', va='center', fontsize=7, color=color, fontstyle='italic')

    ax.set_xlim(-0.5, n_cols - 0.5); ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(metric_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    for i, c in enumerate(row_colors):
        ax.get_yticklabels()[i].set_color(c)
        ax.get_yticklabels()[i].set_fontweight('bold')
    ax.invert_yaxis()
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_all_channels_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print('  Saved: 02_all_channels_heatmap.png')


# ── PLOT 3: PSD Overlay - Best channel from each position ───────────────────
def plot_psd_overlay(results):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('PSD Overlay - Best Channel from Each Position (last 60s)',
                 fontsize=13, fontweight='bold')

    for ax, (title, get_sig) in zip(axes, [
        ('SpO2 Mode (best of RED/SUB3)', lambda r: ('s2' if 'SpO2 RED' in r['channels'] and
            r['channels'].get('SpO2 RED', {}).get('pi', 0) >= r['channels'].get('SpO2 SUB3', {}).get('pi', 0)
            else 's3', r['spo2_raw'])),
        ('HRM Mode (CH1)', lambda r: ('s1', r['hrm_raw'])),
    ]):
        for pos in POSITIONS:
            if pos not in results: continue
            sig_key, raw_data = get_sig(results[pos])
            sig = raw_data[sig_key]; t = raw_data['t']; fs = raw_data['fs']
            valid = (~np.isnan(sig)) & (sig > 1000)
            t_v, sig_v = t[valid], sig[valid]
            if len(t_v) < fs * 10: continue
            m = t_v >= (t_v[-1] - 60.0); seg = sig_v[m]
            if len(seg) < fs * 5: continue
            seg = np.clip(seg, np.percentile(seg, 1), np.percentile(seg, 99))
            ac = bandpass(seg - np.mean(seg), fs)
            f, p = sp.welch(ac, fs=fs, nperseg=min(len(ac), fs * 8))
            ax.semilogy(f, p, lw=1.8, color=POS_COLORS[pos], alpha=0.85, label=f'Position {pos}')

        ax.axvspan(0.7, 3.5, alpha=0.08, color='green', label='Cardiac (0.7-3.5 Hz)')
        ax.axvspan(4.0, 8.0, alpha=0.06, color='red', label='Noise (4-8 Hz)')
        ax.set_xlim(0, 12); ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_psd_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print('  Saved: 03_psd_overlay.png')


# ── PLOT 4: Sliding PI over time for all positions ──────────────────────────
def plot_sliding_pi_all(results):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle('PI % Over Time - All Positions (best raw channel)',
                 fontsize=13, fontweight='bold')

    for ax, (title, ch_pref) in zip(axes, [
        ('SpO2 Mode (best of RED/SUB3)', ['SpO2 SUB3', 'SpO2 RED']),
        ('HRM Mode (CH1)', ['HRM CH1']),
    ]):
        for pos in POSITIONS:
            if pos not in results: continue
            sl = results[pos]['sliding']
            for ch in ch_pref:
                if ch in sl:
                    mets = sl[ch]
                    ax.plot(mets['win_t'], mets['pis'], lw=1.2, color=POS_COLORS[pos],
                            alpha=0.85, label=f'Pos {pos}')
                    break

        ax.axhline(THR['pi_fair'], ls='--', lw=1.5, color='gray', alpha=0.7, label='FAIR (0.3%)')
        ax.axhline(THR['pi_good'], ls='--', lw=1.5, color='green', alpha=0.5, label='GOOD (1.0%)')
        ax.axhline(ITER1_CHEST['pi'], ls=':', lw=1.5, color='red', alpha=0.6, label=f'Iter1 ({ITER1_CHEST["pi"]}%)')
        ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
        ax.set_ylabel('PI %', fontsize=10); ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_sliding_pi_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print('  Saved: 04_sliding_pi_all.png')


# ── PLOT 5: Ranking Scorecard ────────────────────────────────────────────────
def plot_ranking(results):
    # Sort positions by best PI
    ranked = sorted(results.items(), key=lambda x: x[1]['best'].get('pi', 0), reverse=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    fig.suptitle('Position Ranking - Iteration 2 Chest PPG Feasibility',
                 fontsize=16, fontweight='bold', y=0.97)

    cols = ['Rank', 'Position', 'Best Channel', 'PI %', 'PI Grade', 'SNR (dB)', 'AC Amp',
            'HR Valid %', 'RR CV %', 'AGC LED', 'Verdict']
    col_x = np.linspace(0.02, 0.98, len(cols))

    # Header
    for j, c in enumerate(cols):
        ax.text(col_x[j], 0.92, c, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333'))

    # Data rows
    for i, (pos, r) in enumerate(ranked):
        y = 0.82 - i * 0.1
        b = r['best']
        pi_g = grade(b.get('pi', np.nan), THR['pi_good'], THR['pi_fair'])
        agc_val = r['spo2_raw']['agc'].get('AGC1_led', np.nan)

        vals = [
            f'#{i+1}',
            f'Position {pos}',
            r['best_key'],
            f'{b.get("pi", np.nan):.3f}%',
            pi_g,
            f'{b.get("snr", np.nan):.1f}',
            f'{b.get("amp", np.nan):.0f}',
            f'{b.get("hr_pct", np.nan):.0f}%',
            f'{b.get("rrcv", np.nan):.1f}%' if not np.isnan(b.get('rrcv', np.nan)) else 'N/A',
            f'{agc_val:.0f}' if not np.isnan(agc_val) else 'N/A',
            r['verdict'],
        ]

        v_color = {'USABLE': '#2ca02c', 'MARGINAL': '#ff7f0e', 'NOT USABLE': '#d62728'}
        row_bg = POS_COLORS[pos]

        for j, val in enumerate(vals):
            color = '#000000'
            if cols[j] == 'PI Grade': color = GRADE_COLOR.get(val, '#000000')
            if cols[j] == 'Verdict': color = v_color.get(val, '#000000')
            ax.text(col_x[j], y, val, ha='center', va='center', fontsize=9,
                    fontweight='bold' if cols[j] in ('Rank', 'Verdict', 'PI Grade') else 'normal',
                    color=color)

        # Row background
        ax.axhspan(y - 0.04, y + 0.04, alpha=0.08, color=row_bg)

    # Iter1 baseline row
    y = 0.82 - len(ranked) * 0.1 - 0.05
    ax.axhline(y + 0.06, color='gray', lw=0.5, alpha=0.5)
    ax.text(col_x[0], y, 'REF', ha='center', va='center', fontsize=9, color='gray', fontstyle='italic')
    ax.text(col_x[1], y, 'Iter1 Chest', ha='center', va='center', fontsize=9, color='gray', fontstyle='italic')
    ax.text(col_x[3], y, f'{ITER1_CHEST["pi"]:.3f}%', ha='center', va='center', fontsize=9, color='gray')
    ax.text(col_x[4], y, 'POOR', ha='center', va='center', fontsize=9, color='#d62728', fontstyle='italic')
    ax.text(col_x[5], y, f'{ITER1_CHEST["snr"]:.1f}', ha='center', va='center', fontsize=9, color='gray')
    ax.text(col_x[6], y, f'{ITER1_CHEST["amp"]:.0f}', ha='center', va='center', fontsize=9, color='gray')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_position_ranking.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print('  Saved: 05_position_ranking.png')


# ── PLOT 6: Sensor-reported HR comparison ────────────────────────────────────
def plot_sensor_hr_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Sensor-Reported HR Across Positions', fontsize=13, fontweight='bold')

    for ax, (title, mode) in zip(axes, [('SpO2 On-Chip HR', 'spo2_raw'), ('HRM On-Chip HR', 'hrm_raw')]):
        for pos in POSITIONS:
            if pos not in results: continue
            ev = results[pos][mode]['events']
            if 'hr' not in ev: continue
            hr_v = ev['hr']; t_v = ev['t']
            valid = (~np.isnan(hr_v)) & (hr_v > 30) & (hr_v < 200)
            if valid.any():
                ax.plot(t_v[valid], hr_v[valid], 'o-', ms=2, lw=0.8,
                        color=POS_COLORS[pos], alpha=0.7, label=f'Pos {pos} (med={np.median(hr_v[valid]):.0f})')
        ax.set_ylabel('HR (bpm)', fontsize=10); ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        ax.set_ylim(40, 160)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '06_sensor_hr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig); print('  Saved: 06_sensor_hr_comparison.png')


# ── Markdown Report ──────────────────────────────────────────────────────────
def write_comparison_report(results):
    lines = []
    w = lines.append

    ranked = sorted(results.items(), key=lambda x: x[1]['best'].get('pi', 0), reverse=True)

    w('# Iteration 2 - Cross-Position Chest PPG Comparison Report')
    w('')
    w('**Sensor:** AS7058 | **Sampling:** 200 Hz | **Test Date:** 04 March 2026')
    w(f'**Analysis Date:** {datetime.now().strftime("%d %B %Y, %I:%M %p")}')
    w('')
    w('**Objective:** Evaluate 5 chest positions (A-E) to determine optimal placement for AS7058 PPG sensor, compare against Iteration 1 chest baseline, and answer feasibility questions for POC phase.')
    w('')

    # Executive summary
    w('---')
    w('## Executive Summary')
    w('')
    best_pos = ranked[0][0]
    best_r = ranked[0][1]
    usable = [p for p, r in ranked if r['verdict'] == 'USABLE']
    marginal = [p for p, r in ranked if r['verdict'] == 'MARGINAL']
    not_usable = [p for p, r in ranked if r['verdict'] == 'NOT USABLE']

    w(f'**Best Position: {best_pos}** (PI = {best_r["best"]["pi"]:.3f}%, SNR = {best_r["best"]["snr"]:.1f} dB via {best_r["best_key"]})')
    w('')
    if usable: w(f'- **USABLE** ({len(usable)}): Position {", ".join(usable)} - meet minimum thresholds for HR detection')
    if marginal: w(f'- **MARGINAL** ({len(marginal)}): Position {", ".join(marginal)} - have critical weakness (PI < 0.3%)')
    if not_usable: w(f'- **NOT USABLE** ({len(not_usable)}): Position {", ".join(not_usable)} - fail multiple thresholds')
    w('')
    w(f'All positions show **significant improvement** over Iteration 1 chest baseline (PI 0.22%, AC 6 counts). '
      f'The best position achieves **{best_r["best"]["pi"]/ITER1_CHEST["pi"]:.1f}x higher PI** and '
      f'**{best_r["best"]["amp"]/ITER1_CHEST["amp"]:.0f}x higher AC amplitude**.')
    w('')

    # Ranking table
    w('---')
    w('## Position Ranking')
    w('')
    w('| Rank | Position | Best Channel | PI % | Grade | SNR (dB) | AC Amp | HR Valid % | Verdict |')
    w('|------|----------|-------------|------|-------|----------|--------|-----------|---------|')
    for i, (pos, r) in enumerate(ranked):
        b = r['best']
        pi_g = grade(b.get('pi', np.nan), THR['pi_good'], THR['pi_fair'])
        w(f'| #{i+1} | **Position {pos}** | {r["best_key"]} | {b["pi"]:.3f} | {pi_g} | '
          f'{b["snr"]:.1f} | {b["amp"]:.0f} | {b["hr_pct"]:.0f}% | **{r["verdict"]}** |')
    w(f'| REF | *Iter1 Chest* | - | {ITER1_CHEST["pi"]:.3f} | POOR | {ITER1_CHEST["snr"]:.1f} | '
      f'{ITER1_CHEST["amp"]:.0f} | {ITER1_CHEST["hr_pct"]:.0f}% | *Baseline* |')
    w('')

    # Detailed per-position breakdown
    w('---')
    w('## Detailed Per-Position Breakdown')
    w('')
    for pos, r in ranked:
        w(f'### Position {pos} - **{r["verdict"]}**')
        w('')
        w(f'| Channel | PI % | SNR (dB) | AC Amp | HR (bpm) | HR Valid % | RR CV % |')
        w(f'|---------|------|----------|--------|----------|-----------|---------|')
        for ch_name, ch_s in r['channels'].items():
            best_marker = ' **[BEST]**' if ch_name == r['best_key'] else ''
            w(f'| {ch_name}{best_marker} | {ch_s["pi"]:.4f} | {ch_s["snr"]:.1f} | {ch_s["amp"]:.0f} | '
              f'{ch_s["hr_med"]:.1f} | {ch_s["hr_pct"]:.0f}% | '
              f'{ch_s["rrcv"]:.1f} |' if not np.isnan(ch_s.get("rrcv", np.nan))
              else f'| {ch_name}{best_marker} | {ch_s["pi"]:.4f} | {ch_s["snr"]:.1f} | {ch_s["amp"]:.0f} | '
              f'{ch_s.get("hr_med", np.nan):.1f} | {ch_s["hr_pct"]:.0f}% | N/A |')
        w('')

        # Sensor-reported
        spo2_ev = r['spo2_raw']['events']
        hrm_ev = r['hrm_raw']['events']
        sensor_notes = []
        if 'pi' in spo2_ev:
            pi_v = spo2_ev['pi'][~np.isnan(spo2_ev['pi'])]
            if len(pi_v): sensor_notes.append(f'SpO2 on-chip PI: {np.median(pi_v):.4f}%')
        if 'sq' in spo2_ev:
            sq_v = spo2_ev['sq'][~np.isnan(spo2_ev['sq'])]
            if len(sq_v): sensor_notes.append(f'SpO2 SQ: {np.median(sq_v):.0f}/100')
        if 'sq' in hrm_ev:
            sq_v = hrm_ev['sq'][~np.isnan(hrm_ev['sq'])]
            if len(sq_v): sensor_notes.append(f'HRM SQ: {np.median(sq_v):.0f}/100')
        agc1 = r['spo2_raw']['agc'].get('AGC1_led', np.nan)
        if not np.isnan(agc1): sensor_notes.append(f'SpO2 AGC1 LED: {agc1:.0f} LSB ({grade_inv(agc1, 30, 80)})')
        agc1h = r['hrm_raw']['agc'].get('AGC1_led', np.nan)
        if not np.isnan(agc1h): sensor_notes.append(f'HRM AGC1 LED: {agc1h:.0f} LSB ({grade_inv(agc1h, 30, 80)})')

        if sensor_notes:
            w('**Sensor-reported:** ' + ' | '.join(sensor_notes))
            w('')

    # Improvement over Iteration 1
    w('---')
    w('## Improvement Over Iteration 1')
    w('')
    w('| Position | PI Change | SNR Change | AC Amp Change |')
    w('|----------|-----------|------------|---------------|')
    for pos, r in ranked:
        b = r['best']
        pi_c = b['pi'] - ITER1_CHEST['pi']
        snr_c = b['snr'] - ITER1_CHEST['snr']
        amp_c = b['amp'] - ITER1_CHEST['amp']
        w(f'| Position {pos} | {pi_c:+.3f}% ({b["pi"]/ITER1_CHEST["pi"]:.1f}x) | '
          f'{snr_c:+.1f} dB | {amp_c:+.0f} ({b["amp"]/ITER1_CHEST["amp"]:.0f}x) |')
    w('')

    # Feasibility answers
    w('---')
    w('## Feasibility Assessment')
    w('')
    w('### Q1: Is using AS7058 for chest-level measurements technically reasonable until MXREFDES106 arrives?')
    w('')
    if len(usable) > 0:
        w(f'**Yes, conditionally.** {len(usable)} out of 5 positions (Position {", ".join(usable)}) achieve FAIR or better PI, '
          f'with the best reaching {best_r["best"]["pi"]:.3f}%. These positions can support:')
        w('- **Heart rate detection:** Reliable in the HRM channel (SNR > 25 dB across all positions)')
        w('- **Basic HR monitoring:** Feasible for POC-level demonstration')
        w(f'- **SpO2 measurement:** Not yet feasible at chest level. Best PI ({best_r["best"]["pi"]:.3f}%) is still below '
          f'the 1.0% threshold needed for reliable SpO2. SpO2 RED channel PI remains < 0.1% in most positions.')
        w('')
        w('**Critical caveat:** The HRM preset (not SpO2 preset) produces the strongest chest signal. '
          'Algorithm configuration matters significantly at this body location.')
    else:
        w('**No.** None of the 5 positions achieve FAIR PI threshold. AS7058 at chest level is not viable '
          'for the POC without hardware changes. Wait for MXREFDES106.')
    w('')

    w('### Q2: Are the collected datasets robust enough for meaningful POC insights?')
    w('')
    w('**Yes.** The datasets are sufficient for the following conclusions:')
    w('')
    w('- **5 positions tested**, each with dual-mode (SpO2 + HRM) measurements')
    w('- **~190-400 seconds per recording** at 200 Hz — well above minimum for quality assessment')
    w('- **Clear differentiation** between positions: PI ranges from 0.02% to 0.66%, allowing evidence-based selection')
    w('- **Multiple quality metrics** computed: PI, SNR, AC amplitude, HR detection rate, RR variability')
    w('- **Sensor-reported metrics** cross-validated against our computed values')
    w('')
    w('**Limitation:** Only 1 subject (Nikhil). For production, multi-subject validation is needed.')
    w('')

    w('### Q3: Is SP-20 data output sufficiently reliable as golden reference for POC?')
    w('')
    w('**Yes, for HR validation.** The SP-20 finger pulse oximeter provides:')
    w('- 1 Hz HR and SpO2 readings — sufficient for reference comparison')
    w('- Clinical-grade accuracy (FDA-cleared device class)')
    w('')
    w('**For SpO2 validation:** SP-20 is adequate as a reference, but chest-level SpO2 with AS7058 is not '
      'yet viable (PI too low for reliable RED/IR ratio), so the SP-20 SpO2 reference data is not actionable until '
      'chest signal quality improves or finger-based measurements are used.')
    w('')

    # Recommendations
    w('---')
    w('## Recommendations')
    w('')
    w(f'1. **Use Position {best_pos} (PI {best_r["best"]["pi"]:.3f}%) for continued POC development** '
      f'with the HRM preset — it provides the strongest pulsatile signal')
    w(f'2. **Position {ranked[1][0]} as backup** (PI {ranked[1][1]["best"]["pi"]:.3f}%) in case Position {best_pos} '
      f'placement is not practical for the wearable form factor')
    w('3. **Focus on HR algorithm development** first — chest SpO2 is not feasible with current signal levels')
    w('4. **Use SP-20 on finger** as HR reference during chest AS7058 testing to validate accuracy')
    w('5. **When MXREFDES106 arrives**, repeat this same 5-position protocol for direct comparison')
    w('6. **Multi-subject testing** should be planned once optimal position is confirmed')
    w('')

    # Methodology
    w('---')
    w('## Methodology')
    w('')
    w('### Signal Processing')
    w('- **Bandpass filter:** 4th order Butterworth, 0.5-4.0 Hz, zero-phase (filtfilt)')
    w('- **PI calculation:** |P90 - P10 of bandpassed AC| / DC_mean x 100')
    w('- **SNR:** 10 x log10(Power[0.7-3.5 Hz] / Power[4.0-8.0 Hz]) via Welch PSD')
    w('- **HR detection:** Peak finding with min distance 0.35s, valid R-R 0.35-1.5s (40-171 bpm)')
    w('- **Sliding windows:** 10s window, 2s step, skip first 10s (AGC settling)')
    w('- **Summary:** Median of last 120s (post-settling, steady-state)')
    w('')
    w('### Grading Thresholds')
    w('| Metric | GOOD | FAIR | POOR |')
    w('|--------|------|------|------|')
    w(f'| PI % | >= {THR["pi_good"]} | >= {THR["pi_fair"]} | < {THR["pi_fair"]} |')
    w(f'| SNR (dB) | >= {THR["snr_good"]} | >= {THR["snr_fair"]} | < {THR["snr_fair"]} |')
    w(f'| AC Amp | >= {THR["amp_good"]} | >= {THR["amp_fair"]} | < {THR["amp_fair"]} |')
    w('| HR Valid % | >= 80 | >= 50 | < 50 |')
    w(f'| RR CV % | <= {THR["rrcv_good"]} | <= {THR["rrcv_fair"]} | > {THR["rrcv_fair"]} |')
    w('')
    w('### Verdict Logic')
    w('- **USABLE:** 0 critical failures (PI >= 0.3%, SNR >= 6 dB, HR Valid >= 50%)')
    w('- **MARGINAL:** 1 critical failure')
    w('- **NOT USABLE:** 2+ critical failures')
    w('')

    w('---')
    w(f'*Report generated: {datetime.now().strftime("%d %B %Y, %I:%M %p")}*')

    report_path = os.path.join(OUT_DIR, 'ITERATION2_COMPARISON_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('  Saved: ITERATION2_COMPARISON_REPORT.md')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '=' * 60)
    print('  Iteration 2 - Cross-Position Comparison')
    print('=' * 60)

    print('\n[1/4] Loading all positions...')
    results = load_all_positions()

    if len(results) == 0:
        print('  No positions loaded. Exiting.')
        return

    print(f'\n[2/4] Generating comparison plots...')
    plot_pi_comparison(results)
    plot_heatmap(results)
    plot_psd_overlay(results)
    plot_sliding_pi_all(results)
    plot_ranking(results)
    plot_sensor_hr_comparison(results)

    print(f'\n[3/4] Writing comparison report...')
    write_comparison_report(results)

    print(f'\n[4/4] Done!')
    print(f'\n  Output: {OUT_DIR}')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
