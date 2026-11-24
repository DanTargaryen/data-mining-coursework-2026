#!/usr/bin/env python3
"""Run the cross-modal extension using only normal examples for fit and calibration."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torchvision.models import ResNet18_Weights, resnet18

from images import CATEGORIES, image_paths, load_features
from run import (DEFAULT_DATA, DEFAULT_OUT, SEED, classification_metrics, font_manager,
                 plt, read_thyroid, save_plot, threadpool_limits, write_csv, write_json)

ALPHA = 0.05
TOP_FRACTION = 0.025
REFERENCE_CAP = 8192


@dataclass(frozen=True)
class Detection:
    scores: np.ndarray
    p_values: np.ndarray
    flags: np.ndarray
    fit_n: int
    reference_n: int
    calibration_n: int
    pooled_parts: int
    calibration_scores: np.ndarray


def nearest_scores(samples, bank):
    """Reduce the nearest squared Euclidean distances of the highest-scoring parts."""
    count, parts, dim = samples.shape
    queries = samples.reshape(-1, dim)
    bank_transpose = bank.T.contiguous()
    bank_norm = bank.square().sum(dim=1)
    distances = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), 512):
        raw = np.ascontiguousarray(queries[start:start + 512], dtype=np.float32)
        query = torch.from_numpy(raw)
        squared = query.square().sum(dim=1, keepdim=True) + bank_norm - 2 * (query @ bank_transpose)
        distances[start:start + len(raw)] = squared.amin(dim=1).clamp_min(0).numpy()
    distances = distances.reshape(count, parts)
    keep = max(1, int(np.ceil(parts * TOP_FRACTION)))
    return np.partition(distances, -keep, axis=1)[:, -keep:].mean(axis=1), keep


def rank_p_values(calibration, scores):
    """Ties count as equally extreme, so an identical score cannot lower p."""
    ordered = np.sort(calibration)
    at_least_as_extreme = len(ordered) - np.searchsorted(ordered, scores, side='left')
    return (1 + at_least_as_extreme) / (len(ordered) + 1)


def fit_calibrate_score(fit, calibration, test):
    """Fit and calibrate from feature tensors shaped (records, parts, dimensions)."""
    if not len(fit) or not len(calibration) or not len(test):
        raise ValueError('Fit, calibration, and test partitions must be nonempty')
    flat = fit.reshape(-1, fit.shape[-1])
    if len(flat) > REFERENCE_CAP:
        indices = np.random.default_rng(SEED).choice(len(flat), size=REFERENCE_CAP, replace=False)
        flat = flat[indices]
    bank = torch.from_numpy(np.ascontiguousarray(flat, dtype=np.float32))
    reference_scores, _ = nearest_scores(calibration, bank)
    scores, pooled = nearest_scores(test, bank)
    p_values = rank_p_values(reference_scores, scores)
    return Detection(scores, p_values, (p_values <= ALPHA).astype(int), len(fit), len(flat),
                     len(calibration), pooled, reference_scores)


def evaluate(truth, detection):
    metrics = classification_metrics(truth, detection.scores, detection.flags)
    metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn'])
    return metrics


def experiment_record(labels, detector):
    return {
        'fit_normal_n': detector.fit_n, 'calibration_normal_n': detector.calibration_n,
        'reference_vectors': detector.reference_n, 'pooled_parts': detector.pooled_parts,
        'test_negative_n': int(np.count_nonzero(labels == 0)), 'test_positive_n': int(labels.sum()),
        'test': evaluate(labels, detector),
    }


def image_experiments(path, cache):
    with zipfile.ZipFile(path) as archive:
        paths = image_paths(archive, 'Image_Anomaly_Detection/')
        torch.hub.set_dir(str(cache / 'torch-hub'))
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().cpu()
        features, digest = load_features(path, archive, paths, model, True, cache)
        patches = features['patches']
        parts = [PurePosixPath(name).parts[1:] for name in paths]
        pending = {}
        for category in CATEGORIES:
            train = np.array([i for i, p in enumerate(parts) if p[:2] == (category, 'train')])
            test = np.array([i for i, p in enumerate(parts) if p[:2] == (category, 'test')])
            # Match task 2's partition, then filter its normal subsets by the dataset's train labels.
            train_labels = np.array([p[2] == 'bad' for p in (parts[i] for i in train)])
            fit, calibration = train_test_split(train, test_size=.2, random_state=SEED, stratify=train_labels)
            normal_fit = [i for i in fit if parts[i][2] == 'good']
            normal_cal = [i for i in calibration if parts[i][2] == 'good']
            detector = fit_calibrate_score(patches[normal_fit], patches[normal_cal], patches[test])
            pending[category] = (test, detector, normal_cal)
        # Test annotations enter only the evaluation step.
        doc = json.loads(archive.read('Image_Anomaly_Detection/image_anomaly_labels.json'))
        records = {}
        for category, (test, detector, calibration) in pending.items():
            truth = np.array([int(doc['/'.join(parts[i])]['label'] == 'bad') for i in test])
            records[category] = (truth, detector, [paths[i] for i in test], [paths[i] for i in calibration])
    return records, digest


def thyroid_experiment(path):
    archive = path.read_bytes()
    train, test, truth = read_thyroid(archive)
    _, groups = np.unique(train, axis=0, return_inverse=True)
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    fit, calibration = next(split.split(train, groups=groups))
    if np.intersect1d(groups[fit], groups[calibration]).size:
        raise ValueError('Duplicate feature group leaked across the normal split')
    scale = StandardScaler().fit(train[fit])
    fit_parts = scale.transform(train[fit])[:, None, :]
    cal_parts = scale.transform(train[calibration])[:, None, :]
    test_parts = scale.transform(test)[:, None, :]
    detection = fit_calibrate_score(fit_parts, cal_parts, test_parts)
    train_vectors = set(map(tuple, train))
    overlap = np.array([tuple(row) in train_vectors for row in test])
    not_seen = ~overlap
    sensitivity = classification_metrics(truth[not_seen], detection.scores[not_seen], detection.flags[not_seen])
    record = (truth, detection, list(range(1, len(test) + 1)), (calibration + 1).tolist())
    reduced = {'aggregate': sensitivity, 'excluded_test_rows': int(overlap.sum()), 'remaining_test_rows': int(not_seen.sum())}
    return record, hashlib.sha256(archive).hexdigest(), reduced


def plot_ranks(records, out):
    titles = ('Hazelnut (image)', 'Zipper (image)', 'Thyroid (tabular)')
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), sharey=True, layout='constrained')
    for (truth, detection, *_), title, ax in zip(records.values(), titles, axes):
        for label, color, desc in [(0, '#146491', 'Normal'), (1, '#a84d22', 'Anomaly')]:
            p = np.sort(detection.p_values[truth == label])
            fraction = np.arange(1, len(p) + 1) / len(p)
            ax.step(np.r_[p[0], p, 1], np.r_[0, fraction, 1], where='post', color=color, label=desc)
        ax.axvline(ALPHA, ls='--', lw=.9, color='.35', label='Alert threshold')
        ax.set(xscale='log', xlim=(.002, 1.04), ylim=(0, 1.04), title=title, xlabel='Calibration p-value')
        ax.grid(alpha=.22)
    axes[0].set_ylabel('Fraction at or below p')
    axes[-1].legend(loc='lower right', fontsize=7)
    save_plot(fig, out / 'extension-ranks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--cache', type=Path, default=Path(__file__).resolve().parent / '.image-cache')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--figure-font', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    family = 'STIXGeneral'
    if args.figure_font:
        font_manager.fontManager.addfont(args.figure_font)
        family = font_manager.FontProperties(fname=args.figure_font).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 8, 'pdf.fonttype': 42})
    with threadpool_limits(limits=1):
        records, image_sha = image_experiments(args.data / 'Image_Anomaly_Detection.zip', args.cache)
        records['thyroid'], thyroid_sha, sensitivity = thyroid_experiment(args.data / 'thyroid.zip')
    metrics = {key: experiment_record(truth, detection) for key, (truth, detection, *_) in records.items()}
    summary = {
        'seed': SEED, 'alpha': ALPHA, 'top_fraction': TOP_FRACTION, 'reference_cap': REFERENCE_CAP,
        'scoring': 'mean top ceil(0.025 * parts) of nearest squared Euclidean distances',
        'dataset_sha256': {'image': image_sha, 'thyroid': thyroid_sha}, 'groups': metrics,
        'thyroid_without_identical_train_vectors': sensitivity,
    }
    write_json(args.out / 'extension-metrics.json', summary)
    write_csv(args.out / 'extension-predictions.csv', ['group', 'sample_id', 'label', 'distance_score', 'p_value', 'prediction'],
              ((key, ident, int(y), float(score), float(p), int(flag))
               for key, (truth, det, ids, _) in records.items()
               for ident, y, score, p, flag in zip(ids, truth, det.scores, det.p_values, det.flags)))
    write_csv(args.out / 'extension-calibration.csv', ['group', 'sample_id', 'distance_score'],
              ((key, ident, float(score)) for key, (_, det, _, ids) in records.items()
               for ident, score in zip(ids, det.calibration_scores)))
    plot_ranks(records, args.out)
    for key, result in metrics.items():
        print(key, json.dumps(result['test']))


if __name__ == '__main__':
    main()
