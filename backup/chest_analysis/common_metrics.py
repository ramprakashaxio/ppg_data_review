"""
Shared signal quality functions for all chest analysis scripts.
Imported by V1_Raw, V1_Filtered, V2_Raw, V2_Filtered analyze.py scripts.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal as sp

warnings.filterwarnings('ignore')

# ── Quality thresholds ────────────────────────────────────────────────────────
THR = {
    'pi_good':    1.0,   # PI%  >= 1.0  → GOOD
    'pi_fair':    0.3,   # PI%  >= 0.3  → FAIR
    'snr_good':  10.0,   # dB   >= 10   → SpO2 reliable
    'snr_fair':   6.0,   # dB   >= 6    → HR usable
    'rrcv_good': 10.0,   # %    < 10    → clean rhythm
    'rrcv_fair': 25.0,   # %    < 25    → usable
    'hr_lo':     40.0,
    'hr_hi':    160.0,
}

# ── Loading ───────────────────────────────────────────────────────────────────
def load_file(filepath):
    """
    Load a chest CSV (raw or filtered), detect schema, return clean dict.

    Returns
    -------
    dict with keys:
        t       : timestamps reset to 0 (seconds)
        ir      : PPG1_SUB1 values (IR channel)
        red     : PPG1_SUB2 values (RED channel)
        fs      : sampling rate (Hz)
        ftype   : 'raw' or 'filtered'
        agc     : settled AGC1 current (NaN if unavailable)
        acc_t   : accelerometer timestamps (empty array if unavailable)
        acc_mag : accelerometer magnitude  (empty array if unavailable)
        pi_sensor  : sensor-reported PI% series (empty if unavailable)
        hr_sensor  : sensor-reported HR series  (empty if unavailable)
        sq_sensor  : sensor signal quality series (empty if unavailable)
        duration   : total recording length (s)
    """
    raw = pd.read_csv(filepath, low_memory=False)

    ts  = pd.to_numeric(raw['TIMESTAMP [s]'], errors='coerce')
    s1  = pd.to_numeric(raw['PPG1_SUB1'],     errors='coerce')
    s2  = pd.to_numeric(raw['PPG1_SUB2'],     errors='coerce')

    # Detect type: raw files have large DC (~300k+ ADC), filtered have AC (~±50)
    s1_nonna = s1.dropna()
    ftype = 'raw' if s1_nonna.median() > 1000 else 'filtered'

    if ftype == 'raw':
        ppg_mask = s1.notna() & (s1 > 1000)
    else:
        ppg_mask = s1.notna()

    ppg = raw[ppg_mask].copy()
    t   = pd.to_numeric(ppg['TIMESTAMP [s]'], errors='coerce').values
    ir  = pd.to_numeric(ppg['PPG1_SUB1'],     errors='coerce').values
    red = pd.to_numeric(ppg['PPG1_SUB2'],     errors='coerce').values

    order = np.argsort(t)
    t, ir, red = t[order], ir[order], red[order]
    t = t - t[0]

    dt = np.diff(t)
    fs = int(round(1.0 / np.median(dt[dt > 0])))

    # AGC
    agc_cols = [c for c in raw.columns if 'AGC1' in c and 'CURRENT' in c]
    if agc_cols:
        agc_s = pd.to_numeric(raw[agc_cols[0]], errors='coerce').dropna()
        agc   = float(agc_s.iloc[-1]) if len(agc_s) else np.nan
    else:
        agc = np.nan

    # Accelerometer
    if 'ACC_X' in raw.columns:
        ax_ = pd.to_numeric(raw['ACC_X'], errors='coerce')
        ay_ = pd.to_numeric(raw['ACC_Y'], errors='coerce')
        az_ = pd.to_numeric(raw['ACC_Z'], errors='coerce')
        acc_mask = ax_.notna()
        acc_rows = raw[acc_mask]
        at  = pd.to_numeric(acc_rows['TIMESTAMP [s]'], errors='coerce').values
        amg = np.sqrt(
            pd.to_numeric(acc_rows['ACC_X'], errors='coerce').values**2 +
            pd.to_numeric(acc_rows['ACC_Y'], errors='coerce').values**2 +
            pd.to_numeric(acc_rows['ACC_Z'], errors='coerce').values**2
        )
        acc_t   = at - at[0] if len(at) else np.array([])
        acc_mag = amg
    else:
        acc_t, acc_mag = np.array([]), np.array([])

    # Sensor events (only available in raw files that have the SPO2 columns)
    sq_col = 'SPO2: SIGNAL_QUALITY'
    if sq_col in raw.columns:
        sq_mask = pd.to_numeric(raw[sq_col], errors='coerce').notna()
        ev = raw[sq_mask].copy()
    else:
        ev = pd.DataFrame()

    def ev_series(col):
        if col in ev.columns:
            vals = pd.to_numeric(ev[col], errors='coerce')
            evts = pd.to_numeric(ev['TIMESTAMP [s]'], errors='coerce')
            t0   = evts.iloc[0] if len(evts) else 0
            return evts.values - float(t0), vals.values
        return np.array([]), np.array([])

    sq_t,  sq_v  = ev_series(sq_col)
    pi_t,  pi_v  = ev_series('SPO2: PI [%]')
    hr_t,  hr_v  = ev_series('SPO2: HEART_RATE [bpm]')

    return dict(
        t=t, ir=ir, red=red, fs=fs, ftype=ftype, agc=agc,
        acc_t=acc_t, acc_mag=acc_mag,
        pi_sensor=(pi_t, pi_v), hr_sensor=(hr_t, hr_v),
        sq_sensor=(sq_t, sq_v),
        duration=float(t[-1]),
    )


# ── Signal processing ─────────────────────────────────────────────────────────
def bandpass(sig, fs, lo=0.5, hi=4.0, order=4):
    nyq = fs / 2.0
    b, a = sp.butter(order, [lo/nyq, min(hi/nyq, 0.99)], btype='band')
    return sp.filtfilt(b, a, sig)


def compute_snr(ac_sig, fs):
    f, p = sp.welch(np.nan_to_num(ac_sig - np.mean(ac_sig)),
                    fs=fs, nperseg=min(len(ac_sig), int(fs*8)))
    s = np.trapz(p[(f>=0.7)&(f<=3.5)], f[(f>=0.7)&(f<=3.5)])
    n = np.trapz(p[(f>=4.0)&(f<=8.0)], f[(f>=4.0)&(f<=8.0)])
    return 10*np.log10(s/n) if s>0 and n>0 else 0.0


def detect_beats(ac_sig, ts_seg, fs):
    dist    = int(fs * 0.35)
    thr     = np.nanmedian(ac_sig) + 0.35*np.nanstd(ac_sig)
    peaks, _ = sp.find_peaks(ac_sig, distance=dist, height=thr)
    if len(peaks) < 2:
        return 0.0, 99.0
    rr = np.diff(ts_seg[peaks])
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) == 0:
        return 0.0, 99.0
    hr   = 60.0 / np.median(rr)
    rrcv = np.std(rr) / np.mean(rr) * 100 if np.mean(rr) > 0 else 99.0
    return float(hr), float(rrcv)


# ── Sliding window metrics ────────────────────────────────────────────────────
def compute_sliding_metrics(ds, win_s=10.0, step_s=2.0, skip_s=5.0):
    """
    Compute all 6 quality metrics in sliding windows.

    Returns dict of np.arrays aligned to win_t (window centre times).
    """
    t, ir, fs, ftype = ds['t'], ds['ir'], ds['fs'], ds['ftype']
    dur = ds['duration']

    win_t, pis, snrs, ac_amps, hrs, rrcvs = [], [], [], [], [], []

    start = skip_s
    while start + win_s <= dur:
        end = start + win_s
        m   = (t >= start) & (t < end)
        seg = ir[m].copy()
        ts_s = t[m]

        if len(seg) < fs * 3:
            start += step_s
            continue

        # Clip outliers
        p1, p99 = np.percentile(seg, 1), np.percentile(seg, 99)
        seg = np.clip(seg, p1, p99)

        if ftype == 'raw':
            ac = bandpass(seg - np.mean(seg), fs)
            dc = np.mean(seg)
            pi = abs(np.percentile(ac, 90) - np.percentile(ac, 10)) / abs(dc) * 100 if dc > 0 else 0.0
        else:
            ac = seg
            pi = np.nan   # no DC in filtered files

        ac_ptp = abs(np.percentile(ac, 90) - np.percentile(ac, 10))
        snr    = compute_snr(ac, fs)
        hr, rrcv = detect_beats(ac, ts_s, fs)

        win_t.append(start + win_s/2)
        pis.append(pi)
        snrs.append(snr)
        ac_amps.append(ac_ptp)
        hrs.append(hr)
        rrcvs.append(rrcv)
        start += step_s

    return dict(
        win_t   = np.array(win_t,   dtype=float),
        pis     = np.array(pis,     dtype=float),
        snrs    = np.array(snrs,    dtype=float),
        ac_amps = np.array(ac_amps, dtype=float),
        hrs     = np.array(hrs,     dtype=float),
        rrcvs   = np.array(rrcvs,   dtype=float),
    )


# ── Quality colour coding ─────────────────────────────────────────────────────
def quality_color(val, metric):
    """Return colour string for a scalar metric value."""
    if metric == 'pi':
        if val >= THR['pi_good']:  return '#2E7D32'
        if val >= THR['pi_fair']:  return '#F57F17'
        return '#B71C1C'
    if metric == 'snr':
        if val >= THR['snr_good']: return '#2E7D32'
        if val >= THR['snr_fair']: return '#F57F17'
        return '#B71C1C'
    if metric == 'rrcv':
        if val <= THR['rrcv_good']: return '#2E7D32'
        if val <= THR['rrcv_fair']: return '#F57F17'
        return '#B71C1C'
    return '#1565C0'


# ── Six-metric dashboard plot ─────────────────────────────────────────────────
def plot_six_metrics(ds, metrics, title, out_path):
    """
    6-panel time-series dashboard, one panel per metric.
    Saves PNG to out_path.
    """
    wt   = metrics['win_t']
    pis  = metrics['pis']
    snrs = metrics['snrs']
    amps = metrics['ac_amps']
    hrs  = metrics['hrs']
    rvs  = metrics['rrcvs']
    agc  = ds['agc']
    ftype = ds['ftype']

    valid_hr = (hrs > THR['hr_lo']) & (hrs < THR['hr_hi'])

    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(f'{title}\nAll 6 Quality Metrics Over Time  |  window=10s, step=2s',
                 fontsize=13, fontweight='bold')

    def shade(ax, good_lo=None, good_hi=None, fair_lo=None, fair_hi=None):
        if good_lo is not None:
            ax.axhline(good_lo, color='#2E7D32', lw=1.4, ls='--', alpha=0.8, label=f'GOOD ≥{good_lo}')
        if fair_lo is not None:
            ax.axhline(fair_lo, color='#F57F17', lw=1.2, ls='--', alpha=0.8, label=f'FAIR ≥{fair_lo}')

    # ── 1. PI % ───────────────────────────────────────────────────────────────
    ax = axes[0]
    if ftype == 'raw' and not np.all(np.isnan(pis)):
        ax.fill_between(wt, 0, pis, alpha=0.25, color='#B71C1C')
        ax.plot(wt, pis, color='#B71C1C', lw=1.5, label='Computed PI%')
        pi_t, pi_v = ds['pi_sensor']
        if len(pi_t):
            valid = ~np.isnan(pi_v)
            ax.plot(pi_t[valid], pi_v[valid], 'k.', ms=2.5, alpha=0.5, label='Sensor PI%')
        shade(ax, good_lo=THR['pi_good'], fair_lo=THR['pi_fair'])
        ax.set_ylim(0, max(np.nanpercentile(pis, 98)*1.5, 1.2))
        med = np.nanmedian(pis)
        verdict = 'GOOD' if med>=THR['pi_good'] else 'FAIR' if med>=THR['pi_fair'] else 'POOR'
    else:
        ax.text(0.5, 0.5, 'PI% not available\n(filtered file — DC baseline removed)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray', style='italic')
        verdict = 'N/A'
    ax.set_ylabel('PI %', fontsize=9)
    ax.set_title(f'1. Perfusion Index (PI%)   Median = {np.nanmedian(pis):.3f}%   [{verdict}]',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    # ── 2. SNR ────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(wt, snrs, color='#E65100', lw=1.5, label='SNR')
    ax.fill_between(wt, THR['snr_fair'], snrs,
                    where=snrs >= THR['snr_fair'], alpha=0.2, color='green')
    shade(ax, good_lo=THR['snr_good'], fair_lo=THR['snr_fair'])
    med = np.nanmedian(snrs)
    verdict = 'GOOD' if med>=THR['snr_good'] else 'FAIR' if med>=THR['snr_fair'] else 'POOR'
    ax.set_ylabel('SNR (dB)', fontsize=9)
    ax.set_title(f'2. Signal-to-Noise Ratio   Median = {med:.1f} dB   [{verdict}]',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    # ── 3. AC Amplitude ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(wt, amps, color='#4A148C', lw=1.5, label='AC Amplitude')
    ax.fill_between(wt, 0, amps, alpha=0.18, color='#4A148C')
    med_amp = np.nanmedian(amps)
    ax.axhline(med_amp, color='#4A148C', lw=1, ls=':', alpha=0.6, label=f'Median={med_amp:.1f}')
    ax.set_ylabel('AC (ADC counts)', fontsize=9)
    ax.set_title(f'3. AC Amplitude (pulsatile size)   Median = {med_amp:.1f} counts',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)
    ax.set_ylim(0, np.nanpercentile(amps, 98)*1.4)

    # ── 4. HR ─────────────────────────────────────────────────────────────────
    ax = axes[3]
    ax.scatter(wt[valid_hr], hrs[valid_hr], color='#006064', s=15, alpha=0.7, label='HR detected')
    ax.scatter(wt[~valid_hr], hrs[~valid_hr], color='#B71C1C', s=15, alpha=0.4,
               marker='x', label='Out of range / undetected')
    # Sensor HR
    hr_t, hr_v = ds['hr_sensor']
    if len(hr_t):
        valid = ~np.isnan(hr_v) & (hr_v > 30) & (hr_v < 200)
        ax.plot(hr_t[valid], hr_v[valid], 'k-', lw=1, alpha=0.4, label='Sensor HR')
    ax.axhline(60,  color='gray', lw=0.7, ls=':')
    ax.axhline(100, color='gray', lw=0.7, ls=':')
    pct_valid = valid_hr.mean() * 100
    verdict   = 'GOOD' if pct_valid>=80 else 'FAIR' if pct_valid>=50 else 'POOR'
    ax.set_ylim(20, 180)
    ax.set_ylabel('HR (BPM)', fontsize=9)
    ax.set_title(f'4. Heart Rate from Peak Detection   Valid windows = {pct_valid:.0f}%   [{verdict}]',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    # ── 5. RR CV% ─────────────────────────────────────────────────────────────
    ax = axes[4]
    ax.plot(wt[valid_hr], rvs[valid_hr], color='#1A237E', lw=1.4, label='RR CV%')
    ax.fill_between(wt[valid_hr], 0, rvs[valid_hr],
                    where=rvs[valid_hr] > THR['rrcv_fair'], alpha=0.2, color='red',
                    label='High irregularity zone')
    ax.axhline(THR['rrcv_good'], color='#2E7D32', lw=1.4, ls='--', alpha=0.8,
               label=f'GOOD < {THR["rrcv_good"]}%')
    ax.axhline(THR['rrcv_fair'], color='#F57F17', lw=1.2, ls='--', alpha=0.8,
               label=f'FAIR < {THR["rrcv_fair"]}%')
    med_cv = np.nanmedian(rvs[valid_hr]) if valid_hr.any() else np.nan
    verdict = ('GOOD' if med_cv<=THR['rrcv_good'] else
               'FAIR' if med_cv<=THR['rrcv_fair'] else 'POOR') if not np.isnan(med_cv) else 'N/A'
    ax.set_ylim(0, min(np.nanpercentile(rvs[valid_hr], 95)*1.3 if valid_hr.any() else 50, 100))
    ax.set_ylabel('RR CV %', fontsize=9)
    ax.set_title(f'5. RR Interval Regularity (CV%)   Median = {med_cv:.1f}%   [{verdict}]',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    # ── 6. AGC ────────────────────────────────────────────────────────────────
    ax = axes[5]
    if not np.isnan(agc):
        ax.axhline(agc, color='#33691E', lw=2.5, label=f'Settled AGC = {agc:.0f} LSB')
        ax.fill_between([wt[0], wt[-1]], 0, agc, alpha=0.15, color='#33691E')
        # Show AGC over time from raw file if column exists
        verdict = 'GOOD' if agc <= 30 else 'FAIR' if agc <= 80 else 'POOR'
        ax.set_ylim(0, 200)
        ax.set_yticks([0, 16, 30, 80, 160, 200])
        ax.axhline(30,  color='#2E7D32', lw=1.0, ls='--', alpha=0.6, label='GOOD ≤ 30')
        ax.axhline(80,  color='#F57F17', lw=1.0, ls='--', alpha=0.6, label='FAIR ≤ 80')
        ax.axhline(160, color='#B71C1C', lw=1.0, ls='--', alpha=0.6, label='MAX = 160')
        ax.set_title(f'6. AGC LED Current   Settled = {agc:.0f} LSB   [{verdict}]  (lower=better)',
                     fontsize=10, fontweight='bold', loc='left')
    else:
        ax.text(0.5, 0.5, 'AGC not available\n(filtered file — AGC column not present)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray', style='italic')
        ax.set_title('6. AGC LED Current   [N/A for filtered files]',
                     fontsize=10, fontweight='bold', loc='left')
    ax.set_ylabel('AGC (LSB)', fontsize=9)
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Scorecard bar chart ───────────────────────────────────────────────────────
def plot_scorecard(ds, metrics, title, out_path):
    """
    Single-page scorecard: gauge/bar for each metric showing GOOD/FAIR/POOR.
    """
    wt   = metrics['win_t']
    pis  = metrics['pis']
    snrs = metrics['snrs']
    amps = metrics['ac_amps']
    hrs  = metrics['hrs']
    rvs  = metrics['rrcvs']

    valid_hr = (hrs > THR['hr_lo']) & (hrs < THR['hr_hi'])

    # Summary scalars
    pi_med   = np.nanmedian(pis) if ds['ftype']=='raw' else np.nan
    snr_med  = np.nanmedian(snrs)
    amp_med  = np.nanmedian(amps)
    hr_pct   = valid_hr.mean() * 100
    rrcv_med = np.nanmedian(rvs[valid_hr]) if valid_hr.any() else np.nan
    agc_val  = ds['agc']

    def badge(val, metric):
        c = quality_color(val, metric) if not np.isnan(val) else 'gray'
        if metric == 'pi':
            lbl = 'GOOD' if val>=THR['pi_good'] else 'FAIR' if val>=THR['pi_fair'] else 'POOR'
        elif metric == 'snr':
            lbl = 'GOOD' if val>=THR['snr_good'] else 'FAIR' if val>=THR['snr_fair'] else 'POOR'
        elif metric == 'rrcv':
            lbl = 'GOOD' if val<=THR['rrcv_good'] else 'FAIR' if val<=THR['rrcv_fair'] else 'POOR'
        elif metric == 'hrpct':
            lbl = 'GOOD' if val>=80 else 'FAIR' if val>=50 else 'POOR'
            c   = '#2E7D32' if val>=80 else '#F57F17' if val>=50 else '#B71C1C'
        elif metric == 'agc':
            lbl = 'GOOD' if val<=30 else 'FAIR' if val<=80 else 'POOR'
            c   = '#2E7D32' if val<=30 else '#F57F17' if val<=80 else '#B71C1C'
        else:
            lbl = ''
        return lbl, c

    entries = [
        ('PI %',        pi_med,   '%.3f %%', 'pi',    [0, THR['pi_fair'], THR['pi_good'], 5.0],  True),
        ('SNR (dB)',    snr_med,  '%.1f dB',  'snr',   [0, THR['snr_fair'], THR['snr_good'], 25.0], True),
        ('AC Amplitude','—',      None,        None,    None, None),
        ('HR Valid %',  hr_pct,   '%.0f %%',  'hrpct', [0, 50, 80, 100],  True),
        ('RR CV %',     rrcv_med, '%.1f %%',  'rrcv',  [0, THR['rrcv_good'], THR['rrcv_fair'], 60], False),
        ('AGC Current', agc_val,  '%.0f LSB', 'agc',   [0, 30, 80, 200], False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'{title}\nSignal Quality Scorecard', fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for i, (name, val, fmt, metric, scale, hi_good) in enumerate(entries):
        ax = axes[i]

        if val is None or (isinstance(val, float) and np.isnan(val)) or val == '—':
            ax.text(0.5, 0.5, f'{name}\nNot Available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13, color='gray', style='italic')
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.axis('off')
            continue

        lbl, c = badge(val, metric)

        # Horizontal gauge bar
        if scale:
            bar_vals  = [scale[0], val, scale[-1]]
            bar_color = [c, c, '#eeeeee']
            ax.barh([0], [val - scale[0]], left=scale[0], color=c, height=0.5, alpha=0.85)
            ax.barh([0], [scale[-1] - val], left=val, color='#eeeeee', height=0.5)
            for thr_v, thr_lbl in [(scale[1], 'FAIR'), (scale[2], 'GOOD')]:
                ax.axvline(thr_v, color='gray', lw=1.5, ls='--', alpha=0.7)
                ax.text(thr_v, 0.32, thr_lbl, ha='center', fontsize=7, color='gray')
            ax.set_xlim(scale[0], scale[-1])
            ax.set_ylim(-0.5, 0.8)
            ax.set_yticks([])

        # Big value text
        ax.text(0.5, 0.68, fmt % val, transform=ax.transAxes,
                ha='center', va='center', fontsize=26, fontweight='bold', color=c)
        ax.text(0.5, 0.18, lbl, transform=ax.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=c, alpha=0.9))

        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel(name.split('(')[-1].replace(')', '') if '(' in name else '', fontsize=8)
        ax.grid(axis='x', alpha=0.2)

    # AC amplitude: show histogram instead
    ax = axes[2]
    ax.hist(amps, bins=30, color='#4A148C', alpha=0.7, edgecolor='white')
    ax.axvline(np.nanmedian(amps), color='#4A148C', lw=2, ls='--',
               label=f'Median = {np.nanmedian(amps):.1f}')
    ax.set_title('AC Amplitude', fontsize=12, fontweight='bold')
    ax.set_xlabel('ADC counts (10-s window)', fontsize=9)
    ax.set_ylabel('# windows', fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    ax.text(0.97, 0.95, f'Median\n{np.nanmedian(amps):.1f} counts',
            transform=ax.transAxes, ha='right', va='top', fontsize=11, fontweight='bold',
            color='white', bbox=dict(boxstyle='round', facecolor='#4A148C', alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Console summary ───────────────────────────────────────────────────────────
def print_summary(ds, metrics, title):
    wt  = metrics['win_t']
    pis = metrics['pis']
    snrs = metrics['snrs']
    hrs = metrics['hrs']
    rvs = metrics['rrcvs']
    valid_hr = (hrs > THR['hr_lo']) & (hrs < THR['hr_hi'])

    SEP = '=' * 55
    print('\n' + SEP)
    print(title)
    print(SEP)
    print(f'  Duration        : {ds["duration"]:.0f}s')
    print(f'  Sampling rate   : {ds["fs"]} Hz')
    print(f'  File type       : {ds["ftype"]}')
    print(f'  Sliding windows : {len(wt)} x 10s')
    print('-' * 55)
    if ds['ftype'] == 'raw':
        pi_m = np.nanmedian(pis)
        lbl  = 'GOOD' if pi_m>=THR['pi_good'] else 'FAIR' if pi_m>=THR['pi_fair'] else 'POOR'
        print(f'  PI %            : {pi_m:.3f}%   [{lbl}]')
    else:
        print('  PI %            : N/A (filtered file)')
    snr_m = np.nanmedian(snrs)
    lbl = 'GOOD' if snr_m>=THR['snr_good'] else 'FAIR' if snr_m>=THR['snr_fair'] else 'POOR'
    print(f'  SNR             : {snr_m:.1f} dB   [{lbl}]')
    print(f'  AC Amplitude    : {np.nanmedian(metrics["ac_amps"]):.1f} counts (median)')
    hr_pct = valid_hr.mean()*100
    lbl = 'GOOD' if hr_pct>=80 else 'FAIR' if hr_pct>=50 else 'POOR'
    print(f'  HR Valid        : {hr_pct:.0f}%   [{lbl}]')
    if valid_hr.any():
        cv_m = np.nanmedian(rvs[valid_hr])
        lbl  = 'GOOD' if cv_m<=THR['rrcv_good'] else 'FAIR' if cv_m<=THR['rrcv_fair'] else 'POOR'
        print(f'  RR CV %         : {cv_m:.1f}%   [{lbl}]')
    agc = ds['agc']
    if not np.isnan(agc):
        lbl = 'GOOD' if agc<=30 else 'FAIR' if agc<=80 else 'POOR'
        print(f'  AGC Current     : {agc:.0f} LSB   [{lbl}]  (lower=better)')
    else:
        print('  AGC Current     : N/A (filtered file)')
    print(SEP)
