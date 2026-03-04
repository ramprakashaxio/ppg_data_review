"""
Chest V1 Filtered — Signal Quality Analysis
Run: py -3 analyze.py
Output: output/ folder next to this script
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common_metrics import load_file, compute_sliding_metrics, \
                           plot_six_metrics, plot_scorecard, print_summary

TITLE    = 'Chest V1 — Filtered File  |  13:18:43'
CSV_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..',
    'AS7058/03_Chest_AS7058/V1_wrist algo/',
    'Chest_position_nikhil_V1_02.032026_2026-03-02_13-18-43_filtered.csv'
)
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

print(f'Loading: {os.path.basename(CSV_PATH)} ...')
ds      = load_file(CSV_PATH)
metrics = compute_sliding_metrics(ds)

print('Plotting 6-metric dashboard ...')
plot_six_metrics(ds, metrics, TITLE,
                 os.path.join(OUT_DIR, '01_six_metrics_dashboard.png'))

print('Plotting scorecard ...')
plot_scorecard(ds, metrics, TITLE,
               os.path.join(OUT_DIR, '02_scorecard.png'))

print_summary(ds, metrics, TITLE)
print(f'\nOutputs saved to: {OUT_DIR}')
