#!/usr/bin/env python3
"""Reproduce the course experiments for weather forecasting and Thyroid novelty detection."""
import argparse
import csv
import hashlib
import io
import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, confusion_matrix, mean_absolute_error,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, root_mean_squared_error)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
HISTORY = 12
CO2_TARGET = 'CO2 (ppm)'
WEATHER_MAIN_MODEL = 'ridge_all'
DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'DM_2026_Dataset'
DEFAULT_OUT = Path(__file__).resolve().parent / 'results'


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, names, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(rows)


def regression_metrics(y, pred):
    return {'mae': float(mean_absolute_error(y, pred)), 'rmse': float(root_mean_squared_error(y, pred))}


def read_weather(raw, target):
    # Only the header uses non-ASCII units. The station uses Latin-1, while
    # the course copy has partially damaged unit strings in some headers.
    first_line = raw.splitlines()[0]
    station_header = first_line.startswith(b'"Date Time"') or first_line.startswith(b'Date Time,')
    encoding = 'latin-1' if station_header else 'utf-8-sig'
    reader = csv.reader(io.StringIO(raw.decode(encoding, errors='replace')))
    header = [name.strip() for name in next(reader)]
    date_formats = {'date': '%Y/%m/%d %H:%M', 'Date Time': '%d.%m.%Y %H:%M:%S'}
    if len(header) != 22 or header[0] not in date_formats:
        raise ValueError('Expected a date column and 21 numeric fields')
    date_format = date_formats[header[0]]
    # The course archive renamed the station's last column from CO2 to OT.
    # Restrict the alias to the verified column layout.
    names = header[1:]
    source_target = target
    if (target == CO2_TARGET and target not in names and names[-2:] == ['Tlog (degC)', 'OT']
            and names[:2] == ['p (mbar)', 'T (degC)']):
        source_target = 'OT'
    if names.count(source_target) != 1:
        raise ValueError(f'Expected exactly one weather target column {target!r}')
    column = names.index(source_target)
    names[column] = target
    records = {}
    duplicate_rows = 0
    for row in reader:
        timestamp = datetime.strptime(row[0].strip(), date_format)
        values = tuple(float(x) for x in row[1:])
        if len(values) != 21:
            raise ValueError(f'Expected 21 features at {timestamp}')
        if timestamp in records:
            if records[timestamp] != values:
                raise ValueError(f'Conflicting readings at {timestamp}')
            duplicate_rows += 1
        records[timestamp] = values
    ordered = sorted(records.items())
    times = np.array([ts for ts, _ in ordered], dtype='datetime64[m]')
    values = np.array([v for _, v in ordered], dtype=np.float64)
    sentinel_counts = np.count_nonzero(values == -9999, axis=0)
    values[values == -9999] = np.nan
    return times, values, names, column, duplicate_rows, sentinel_counts, source_target


def weather_run(csv_path, out, target):
    raw = csv_path.read_bytes()
    times, values, names, col, duplicates, sentinel, source_target = read_weather(raw, target)
    raw_positions = np.arange(HISTORY, len(times))
    gaps = np.diff(times) != np.timedelta64(10, 'm')
    contiguous = ~sliding_window_view(gaps, HISTORY).any(axis=1)
    history = sliding_window_view(values[:-1], HISTORY, axis=0).swapaxes(1, 2)
    history_valid = np.isfinite(history).all(axis=(1, 2))
    labels_valid = np.isfinite(values[HISTORY:, col])
    keep = contiguous & history_valid & labels_valid
    windows, target_position = history[keep], raw_positions[keep]
    y = values[target_position, col]
    last = windows[:, -1, col]
    window_times = times[target_position]
    train_end, validation_end = int(.7 * len(times)), int(.85 * len(times))
    train = target_position < train_end
    val = (target_position >= train_end) & (target_position < validation_end)
    test = target_position >= validation_end
    assert train.any() and val.any() and test.any()
    assert window_times[train][-1] < window_times[val][0] < window_times[test][0]
    np.testing.assert_array_equal(windows[:, -1], values[target_position - 1])

    validation_scores = {'persistence': regression_metrics(y[val], last[val])}
    predictions = {'persistence': last[test]}
    tuning = {}
    for name, features in {'ridge_target': windows[:, :, col], 'ridge_all': windows.reshape(len(windows), -1)}.items():
        models = []
        for alpha in (.1, 1., 10., 100., 1000.):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[train], y[train])
            metrics = regression_metrics(y[val], model.predict(features[val]))
            models.append((alpha, model, metrics))
        alpha, model, scores = min(models, key=lambda item: item[2]['mae'])
        tuning[name] = {'selected_alpha': alpha, 'candidates': [{'alpha': a, **m} for a, _, m in models]}
        validation_scores[name] = scores
        predictions[name] = model.predict(features[test])

    # The nonlinear comparison uses compact summaries of the same 12 observations.
    features = np.column_stack([windows[:, -1, :], windows.mean(axis=1), windows[:, -1, :] - windows[:, 0, :]])
    tree = make_pipeline(StandardScaler(), HistGradientBoostingRegressor(
        max_iter=200, learning_rate=.08, max_leaf_nodes=15, min_samples_leaf=30,
        early_stopping=False, random_state=SEED))
    tree.fit(features[train], y[train])
    validation_scores['boosting'] = regression_metrics(y[val], tree.predict(features[val]))
    predictions['boosting'] = tree.predict(features[test])
    selected = min(validation_scores, key=lambda name: validation_scores[name]['mae'])
    metrics = {name: regression_metrics(y[test], pred) for name, pred in predictions.items()}
    for name, pred in predictions.items():
        delta_metrics = regression_metrics(y[test] - last[test], pred - last[test])
        np.testing.assert_allclose(list(delta_metrics.values()), list(metrics[name].values()), rtol=1e-12)
    rapid_threshold = float(np.quantile(np.abs(y[train] - last[train]), .9))
    rapid = np.abs(y[test] - last[test]) >= rapid_threshold
    stats = {
        'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'target': target,
        'source_target_header': source_target, 'main_model': WEATHER_MAIN_MODEL,
        'history_steps': HISTORY, 'input_variables': len(names),
        'target_range_without_sentinel': [float(np.nanmin(values[:, col])), float(np.nanmax(values[:, col]))],
        'raw_rows': len(times) + duplicates, 'unique_rows': len(times), 'duplicate_rows': duplicates,
        'gap_minutes': np.diff(times)[gaps].astype(int).tolist(),
        'sentinels': {n: int(c) for n, c in zip(names, sentinel) if c},
        'windows': int(keep.sum()), 'split': {
            name: {'n': int(m.sum()), 'first_target': str(window_times[m][0]), 'last_target': str(window_times[m][-1])}
            for name, m in [('train', train), ('validation', val), ('test', test)]},
        'split_boundaries': [str(times[train_end]), str(times[validation_end])],
        'selected_by_validation_mae': selected, 'alpha_tuning': tuning,
        'validation': validation_scores, 'test': metrics,
        'rapid_changes': {'train_p90': rapid_threshold, 'test_n': int(rapid.sum()),
                          'test': {n: regression_metrics(y[test][rapid], p[rapid]) for n, p in predictions.items()}},
        'plotted_windows': min(288, int(test.sum())),
    }
    write_json(out / 'weather-metrics.json', stats)
    write_csv(out / 'weather-predictions.csv', ['target_time', 'previous', 'observed', 'observed_delta',
              *predictions.keys(), *(n + '_delta' for n in predictions)],
              zip(window_times[test], last[test], y[test], y[test] - last[test], *predictions.values(),
                  *(p - last[test] for p in predictions.values())))
    n = stats['plotted_windows']
    index = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.7), sharex=True, layout='constrained')
    axes[0].plot(index, y[test][:n], label='Observed', color='#182432', lw=1.5)
    primary = predictions[WEATHER_MAIN_MODEL][:n]
    axes[0].plot(index, primary, label='All-variable ridge', color='#186ab1', lw=1.0, alpha=.9)
    axes[0].plot(index, last[test][:n], label='Persistence', color='#c06b2a', lw=.9, alpha=.75)
    axes[0].set_ylabel(target)
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(index, y[test][:n] - last[test][:n], color='#182432', lw=1.2)
    axes[1].plot(index, primary - last[test][:n], color='#186ab1', lw=1)
    axes[1].set(xlabel='First valid test windows in time order', ylabel=f'Change in {target}')
    for ax in axes:
        ax.grid(alpha=.22)
    save_plot(fig, out / 'weather-forecast')
    return stats


def save_plot(fig, stem):
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=175, bbox_inches='tight')
    plt.close(fig)


def read_thyroid(archive):
    features = [f'feature_{i}' for i in range(1, 7)]
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        matrices = []
        for split in ('train', 'test'):
            names = features + (['label'] if split == 'test' else [])
            with source.open(f'thyroid/{split}-set.csv') as member:
                reader = csv.DictReader(io.TextIOWrapper(member, encoding='utf-8-sig'))
                if set(reader.fieldnames) != set(names):
                    raise ValueError(f'Unexpected Thyroid {split} columns')
                matrices.append(np.array([[float(r[feature]) for feature in names] for r in reader]))
    train, test = matrices
    if not np.isfinite(train).all() or not np.isfinite(test).all() or set(test[:, -1]) != {0., 1.}:
        raise ValueError('Expected finite features and binary 0/1 test labels')
    return train, test[:, :-1], test[:, -1].astype(int)


def classification_metrics(y, scores, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {'auroc': float(roc_auc_score(y, scores)), 'ap': float(average_precision_score(y, scores)),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
            'accuracy': float(np.mean(y == prediction)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def thyroid_run(data, out):
    archive = (data / 'thyroid.zip').read_bytes()
    train, test, y = read_thyroid(archive)
    _, groups = np.unique(train, axis=0, return_inverse=True)
    fit, calibration = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(train, groups=groups))
    assert not set(groups[fit]) & set(groups[calibration])
    scaler = StandardScaler().fit(train[fit])
    fit_data = scaler.transform(train[fit])
    calibration_data = scaler.transform(train[calibration])
    test_data = scaler.transform(test)
    models = {
        'isolation_forest': IsolationForest(n_estimators=300, max_samples=256, random_state=SEED, n_jobs=1),
        'lof': LocalOutlierFactor(n_neighbors=20, novelty=True),
    }
    zeros = np.zeros_like(y)
    evaluated = {'all_normal': (zeros, zeros)}
    calibrations = {}
    for name, model in models.items():
        model.fit(fit_data)
        calibration_scores = -model.score_samples(calibration_data)
        threshold = float(np.quantile(calibration_scores, .95))
        scores = -model.score_samples(test_data)
        evaluated[name] = (scores, (scores > threshold).astype(int))
        calibrations[name] = {'threshold': threshold, 'fraction_above_threshold': float(np.mean(calibration_scores > threshold))}
    train_vectors = set(map(tuple, train))
    overlaps = np.array([tuple(row) in train_vectors for row in test])
    stats = {
        'dataset_sha256': hashlib.sha256(archive).hexdigest(), 'seed': SEED,
        'train_n': len(train), 'train_unique': len(train_vectors), 'fit_n': len(fit), 'calibration_n': len(calibration),
        'test_n': len(test), 'test_positive_n': int(y.sum()), 'test_prevalence': float(y.mean()),
        'threshold_quantile': .95, 'calibration': calibrations,
        'test': {name: classification_metrics(y, *pair) for name, pair in evaluated.items()},
        'train_test_overlap': {'unique_vectors': len(train_vectors & set(map(tuple, test))),
                               'test_rows': int(overlaps.sum()), 'positive_test_rows': int(y[overlaps].sum())},
        'unmatched_test': {'n': int((~overlaps).sum()), 'positive_n': int(y[~overlaps].sum()),
                           'metrics': {name: classification_metrics(y[~overlaps], pair[0][~overlaps], pair[1][~overlaps])
                                       for name, pair in evaluated.items()}},
    }
    write_json(out / 'thyroid-metrics.json', stats)
    columns = [a for name in models for a in (name + '_score', name + '_prediction')]
    write_csv(out / 'thyroid-predictions.csv', ['test_row_1_based', 'label', 'feature_match_in_train', *columns],
              zip(range(1, len(y) + 1), y, overlaps.astype(int), *(a for name in models for a in evaluated[name])))
    split_rows = sorted([(int(i) + 1, 'fit') for i in fit] + [(int(i) + 1, 'calibration') for i in calibration])
    write_csv(out / 'thyroid-train-split.csv', ['train_row_1_based', 'partition'], split_rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), layout='constrained')
    for name, color in [('isolation_forest', '#186ab1'), ('lof', '#a8551f')]:
        score, _ = evaluated[name]
        precision, recall, _ = precision_recall_curve(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        label = 'Isolation Forest' if name == 'isolation_forest' else 'LOF'
        axes[0].plot(recall, precision, label=f'{label} ({stats["test"][name]["ap"]:.3f})', color=color)
        axes[1].plot(fpr, tpr, label=f'{label} ({stats["test"][name]["auroc"]:.3f})', color=color)
    axes[0].axhline(y.mean(), ls='--', lw=1, color='.42', label=f'Prevalence ({y.mean():.3f})')
    axes[1].plot([0, 1], [0, 1], '--', lw=1, color='.42')
    axes[0].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=.22)
    save_plot(fig, out / 'thyroid-curves')
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('weather', 'thyroid', 'all'), default='all')
    parser.add_argument('--target', choices=(CO2_TARGET, 'OT', 'T (degC)'), default=CO2_TARGET,
                        help='Forecast target (OT is a compatibility alias for CO2)')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--weather-csv', type=Path, help='Read this CSV instead of weather.csv under --data')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--figure-font', type=Path, help='Optional TTF used for plot labels')
    args = parser.parse_args()
    target = CO2_TARGET if args.target == 'OT' else args.target
    args.out.mkdir(parents=True, exist_ok=True)
    if args.figure_font:
        font_manager.fontManager.addfont(args.figure_font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=args.figure_font).get_name()
    else:
        plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    with threadpool_limits(limits=1):
        if args.task in ('weather', 'all'):
            stats = weather_run(args.weather_csv or args.data / 'weather.csv', args.out, target)
            print(f'Weather ({target}) main model: {stats["main_model"]}; '
                  f'lowest validation MAE: {stats["selected_by_validation_mae"]}')
            print(json.dumps(stats['test']))
        if args.task in ('thyroid', 'all'):
            stats = thyroid_run(args.data, args.out)
            print('Thyroid test metrics:')
            print(json.dumps(stats['test']))
    write_json(args.out / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
               'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__,
               'task': args.task, 'target': target, 'seed': SEED, 'figure_font': plt.rcParams['font.family']})


if __name__ == '__main__':
    main()
