#!/usr/bin/env python3
"""Paired time-block bootstrap of fixed, held-out CO2 forecasts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = ('ridge_all', 'ridge_target', 'persistence')
REPLICATES = 10_000
SEED = 42


def signed_differences(totals):
    # Columns: count, three sums of absolute error, three sums of squared error.
    count = totals[..., 0, None]
    mae = totals[..., 1:4] / count
    rmse = np.sqrt(totals[..., 4:] / count)
    return {'mae': mae[..., :1] - mae[..., 1:], 'rmse': rmse[..., :1] - rmse[..., 1:]}


def summarize_block_size(days, mask, errors, block_size):
    # Define blocks by calendar time before filtering any rapid-change subset.
    block_ids = (days - days[0]) // block_size
    count = int(block_ids[-1]) + 1
    blocks = np.zeros((count, 7))
    np.add.at(blocks, block_ids[mask], np.column_stack([np.ones(mask.sum()), np.abs(errors[mask]), errors[mask] ** 2]))
    draws = np.random.default_rng(SEED).integers(count, size=(REPLICATES, count))
    sampled = blocks[draws].sum(axis=1)
    sampled = sampled[sampled[:, 0] > 0]
    baseline = signed_differences(blocks.sum(axis=0))
    samples = signed_differences(sampled)
    result = {}
    for i, name in enumerate(MODELS[1:]):
        result[name] = {}
        for metric in ('mae', 'rmse'):
            lo, hi = np.quantile(samples[metric][:, i], (0.025, 0.975))
            result[name][metric] = {'difference_ppm': float(baseline[metric][i]), 'interval_95_ppm': [float(lo), float(hi)]}
    return {'calendar_blocks': count, 'nonempty_resamples': len(sampled), 'comparisons': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results',
                   help='Directory containing weather-predictions.csv and weather-metrics.json')
    args = p.parse_args()
    metrics = json.loads((args.results/'weather-metrics.json').read_text())
    with (args.results/'weather-predictions.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    ts = np.asarray([r['target_time'] for r in rows], dtype='datetime64[m]')
    if not len(ts) or (np.diff(ts) <= np.timedelta64(0, 'm')).any():
        raise ValueError('Require nonempty, strictly time-ordered held-out forecasts')
    if metrics['target'] != 'CO2 (ppm)' or len(ts) != metrics['split']['test']['n']:
        raise ValueError('Forecasts and run metadata must be from the same CO2 test split')
    actual = np.array([float(r['observed']) for r in rows])
    errors = np.array([[float(r[key]) for key in MODELS] for r in rows]) - actual[:, None]
    rapid = np.abs([float(r['observed_delta']) for r in rows]) >= metrics['rapid_changes']['train_p90']
    if np.count_nonzero(rapid) != metrics['rapid_changes']['test_n']:
        raise ValueError('Rapid-change count disagrees with the training threshold')
    days = ts.astype('datetime64[D]').astype(int)
    output = {'target': metrics['target'], 'difference': 'all-variable minus comparator',
              'resamples': REPLICATES, 'seed': SEED, 'test_n': len(ts), 'rapid_n': int(rapid.sum()),
              'training_change_p90_ppm': metrics['rapid_changes']['train_p90'], 'blocks': {}}
    for size in (1, 2):
        output['blocks'][str(size)] = {
            'all_test': summarize_block_size(days, np.ones(len(days), dtype=bool), errors, size),
            'rapid_test': summarize_block_size(days, rapid, errors, size),
        }
    out = args.results / 'weather-block-bootstrap.json'
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
