#!/usr/bin/env python3
"""Reproduce unsupervised image clustering and image-level anomaly detection."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
import PIL
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, average_precision_score, f1_score,
                             normalized_mutual_info_score, precision_recall_curve,
                             roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import torch
from torch.nn import functional as F
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from run import (DEFAULT_DATA, DEFAULT_OUT, classification_metrics, font_manager,
                 plt, save_plot, sklearn, threadpool_limits, write_csv, write_json)

SEEDS = (7, 23, 42)
CLUSTER_RUNS = ('rgb_kmeans', 'resnet_kmeans', 'resnet_ward')
ANOMALY_RUNS = ('global_logistic', 'local_neighbors')
CATEGORIES = ('hazelnut', 'zipper')
RGB_MEAN = (.485, .456, .406)
RGB_STD = (.229, .224, .225)


def image_paths(archive, prefix):
    return sorted(n for n in archive.namelist() if n.startswith(prefix) and n.lower().endswith('.png') and not PurePosixPath(n).name.startswith('.'))


def read_image(archive, name, size=(224, 224)):
    with archive.open(name) as raw, Image.open(raw) as image:
        return np.asarray(image.convert('RGB').resize(size, Image.Resampling.BILINEAR)).copy()


def rgb_histogram(pixels):
    channels = [np.bincount((pixels[:, :, c] // 16).reshape(-1), minlength=16) for c in range(3)]
    return np.sqrt(np.concatenate(channels) / (3 * pixels.shape[0] * pixels.shape[1]))


def extract_features(archive, paths, model, include_local):
    means = torch.tensor(RGB_MEAN).view(1, 3, 1, 1)
    scales = torch.tensor(RGB_STD).view(1, 3, 1, 1)
    global_features, histogram_features, local_features = [], [], []
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(paths), 24):
            batch = np.stack(list(pool.map(lambda path: read_image(archive, path), paths[start:start + 24])))
            x = (torch.tensor(batch).permute(0, 3, 1, 2).float() / 255 - means) / scales
            stage1 = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            stage2 = model.layer2(stage1)
            stage3 = model.layer3(stage2)
            stage4 = model.layer4(stage3)
            global_features.append(F.normalize(model.avgpool(stage4).flatten(1), dim=1).numpy())
            if include_local:
                layer2 = F.avg_pool2d(stage2, 2)
                layer2 = F.normalize(F.avg_pool2d(layer2, 3, stride=1, padding=1, count_include_pad=False), dim=1)
                layer3 = F.normalize(F.avg_pool2d(stage3, 3, stride=1, padding=1, count_include_pad=False), dim=1)
                patches = F.normalize(torch.cat([layer2, layer3], dim=1), dim=1)
                local_features.append(patches.flatten(2).transpose(1, 2).contiguous().numpy())
            else:
                histogram_features.extend(map(rgb_histogram, batch))
            if start == 0 or start % 144 == 0:
                print(f'Extracted {min(start + len(batch), len(paths))}/{len(paths)} images', flush=True)
    result = {'global_features': np.concatenate(global_features)}
    if include_local:
        result['patches'] = np.concatenate(local_features)
    else:
        result['color'] = normalize(np.stack(histogram_features))
    return result


def load_features(archive_path, archive, paths, model, include_local, cache_dir):
    with archive_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    tag = 'patch' if include_local else 'color'
    cache = cache_dir / f'{tag}-resnet18-v1-{torch.__version__}-{torchvision.__version__}-{digest[:16]}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if data['paths'].tolist() != paths:
                raise ValueError('Cached feature order differs from the archive')
            return {key: data[key] for key in data.files if key != 'paths'}, digest
    features = extract_features(archive, paths, model, include_local)
    temporary = cache.with_suffix('.building.npz')
    np.savez(temporary, paths=np.array(paths), **features)
    temporary.replace(cache)
    return features, digest


def cluster_run(archive_path, out, model, cache):
    with zipfile.ZipFile(archive_path) as archive:
        paths = image_paths(archive, 'Cluster/dataset/')
        if len(paths) != 600:
            raise ValueError(f'Expected 600 clustering images, got {len(paths)}')
        features, digest = load_features(archive_path, archive, paths, model, False, cache)
        assignments = {}
        feature_names = {'rgb_kmeans': 'color', 'resnet_kmeans': 'global_features', 'resnet_ward': 'global_features'}
        for name in CLUSTER_RUNS:
            x = features[feature_names[name]]
            if name == 'resnet_ward':
                assignments[name] = {42: AgglomerativeClustering(n_clusters=6, linkage='ward').fit_predict(x)}
            else:
                assignments[name] = {seed: KMeans(n_clusters=6, n_init=20, random_state=seed).fit_predict(x) for seed in SEEDS}
        # Labels are opened only after all partitions are fixed.
        labels = json.loads(archive.read('Cluster/cluster_labels.json'))
        if set(labels) != {PurePosixPath(n).name for n in paths}:
            raise ValueError('Clustering label set differs from image set')
        y = np.array([labels[PurePosixPath(n).name] for n in paths])
        classes = np.unique(y)
        rows = {}
        scores = {}
        for name in CLUSTER_RUNS:
            x = features[feature_names[name]]
            rows[name] = [{'seed': seed, 'ari': float(adjusted_rand_score(y, z)),
                           'nmi': float(normalized_mutual_info_score(y, z)),
                           'silhouette': float(silhouette_score(x, z))}
                          for seed, z in assignments[name].items()]
            scores[name] = {key: {'mean': float(np.mean([e[key] for e in rows[name]])),
                                  'std': float(np.std([e[key] for e in rows[name]]))}
                            for key in ('ari', 'nmi', 'silhouette')}
        pred = assignments['resnet_kmeans'][42]
        counts = np.array([[np.count_nonzero((y == label) & (pred == c)) for c in range(6)] for label in classes])
        truth_indices, cluster_indices = linear_sum_assignment(-counts)
        cluster_names = dict(zip(cluster_indices.tolist(), classes[truth_indices].tolist()))
        mapped = np.array([cluster_names[k] for k in pred])
        mistakes = sorted([(int(counts[i, c]), label, cluster_names[c]) for i, label in enumerate(classes)
                           for c in range(6) if cluster_names[c] != label and counts[i, c]], reverse=True)
        representative = []
        x = features['global_features']
        for _, actual, destination in mistakes[:3]:
            options = np.flatnonzero((y == actual) & (mapped == destination))
            centroid = x[mapped == destination].mean(axis=0)
            representative.append(int(options[np.argmin(np.linalg.norm(x[options] - centroid, axis=1))]))
        if not representative:
            representative = [int(np.flatnonzero(y == c)[0]) for c in classes[:3]]
        stats = {'dataset_sha256': digest, 'n': len(y), 'class_counts': dict(Counter(y)), 'runs': rows,
                 'metrics': scores, 'shown_run': 'resnet_kmeans_seed42', 'classes': classes.tolist(),
                 'matched_cluster_labels': cluster_names, 'contingency': counts[:, cluster_indices].tolist(),
                 'largest_mismatches': [list(e) for e in mistakes[:5]],
                 'illustrated': [{'path': paths[i], 'true': y[i], 'mapped': mapped[i]} for i in representative]}
        columns = [(name, seed) for name in CLUSTER_RUNS for seed in assignments[name]]
        write_csv(out/'cluster-assignments.csv', ['image', 'true_label', 'resnet_kmeans_42_mapped',
                  *(f'{n}_{s}' for n, s in columns)],
                  zip(paths, y, mapped, *(assignments[n][s] for n, s in columns)))
        write_json(out/'cluster-metrics.json', stats)
        plot_clusters(archive, paths, y, mapped, representative, classes, np.array(stats['contingency']), out)
    return stats


def plot_clusters(archive, paths, y, mapped, examples, classes, matrix, out):
    fig = plt.figure(figsize=(6.4, 3.0), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.15])
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)
    ax.set(xticks=np.arange(len(classes)), xticklabels=classes, yticks=np.arange(len(classes)), yticklabels=classes,
           xlabel='Cluster after evaluation-only matching', ylabel='True category')
    ax.tick_params(axis='x', labelrotation=30)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center', color='white' if value > 55 else '#172235')
    for k, index in enumerate(examples):
        ax = fig.add_subplot(grid[k, 1])
        ax.imshow(read_image(archive, paths[index], (144, 144)))
        ax.axis('off')
        ax.text(1.08, .5, f'True: {y[index]}\nCluster: {mapped[index]}', transform=ax.transAxes, va='center')
    save_plot(fig, out/'cluster-results')


def patch_reference(patches, bank_size):
    flat = patches.reshape(-1, patches.shape[-1])
    sample = np.random.default_rng(42).choice(len(flat), size=min(bank_size, len(flat)), replace=False)
    return torch.from_numpy(np.ascontiguousarray(flat[sample])).T.contiguous()


def local_scores(patches, bank):
    patches = patches.reshape(-1, patches.shape[-1])
    best = []
    for start in range(0, len(patches), 512):
        query = torch.from_numpy(np.ascontiguousarray(patches[start:start+512]))
        distances = 1 - (query @ bank).amax(dim=1)
        best.append(distances.clamp_min(0).numpy())
    distances = np.concatenate(best).reshape(-1, 196)
    return np.partition(distances, -5, axis=1)[:, -5:].mean(axis=1)


def validation_threshold(y, scores):
    candidates = np.unique(scores)
    return float(max(candidates, key=lambda value: (f1_score(y, scores >= value, zero_division=0), value)))


def anomaly_run(archive_path, out, model, cache):
    with zipfile.ZipFile(archive_path) as archive:
        paths = image_paths(archive, 'Image_Anomaly_Detection/')
        if len(paths) != 602:
            raise ValueError(f'Expected 602 detection images, got {len(paths)}')
        features, digest = load_features(archive_path, archive, paths, model, True, cache)
        parts = [PurePosixPath(path).parts[1:] for path in paths]
        staged = {}
        split_records = []
        for category in CATEGORIES:
            train = np.array([i for i,p in enumerate(parts) if p[:2] == (category, 'train')])
            test = np.array([i for i,p in enumerate(parts) if p[:2] == (category, 'test')])
            y_train = np.array([int(parts[i][2] == 'bad') for i in train])
            if len(train) != 250 or np.bincount(y_train).tolist() != [200, 50]:
                raise ValueError(f'Unexpected train class counts for {category}')
            fit, val = train_test_split(train, test_size=.2, random_state=42, stratify=y_train)
            yf = np.array([int(parts[i][2] == 'bad') for i in fit])
            yv = np.array([int(parts[i][2] == 'bad') for i in val])
            x = features['global_features']
            classifier = LogisticRegression(C=100, class_weight='balanced', max_iter=1000)
            classifier.fit(x[fit], yf)
            bank = patch_reference(features['patches'][fit[yf == 0]], 8192)
            validation = {'global_logistic': classifier.decision_function(x[val]),
                          'local_neighbors': local_scores(features['patches'][val], bank)}
            thresholds = {name: validation_threshold(yv, scores) for name, scores in validation.items()}
            vm = {n: classification_metrics(yv, s, s >= thresholds[n]) for n, s in validation.items()}
            staged[category] = (test, thresholds, vm, {
                'global_logistic': classifier.decision_function(x[test]),
                'local_neighbors': local_scores(features['patches'][test], bank),
            })
            split_records.extend((paths[i], category, split, parts[i][2]) for split, ii in [('fit',fit),('validation',val)] for i in sorted(ii))
            print(f'{category}: fit/validation/test={len(fit)}/{len(val)}/{len(test)}', flush=True)
        # Read held-out labels only after the models, thresholds and test scores are fixed.
        labels = json.loads(archive.read('Image_Anomaly_Detection/image_anomaly_labels.json'))
        if set(labels) != {'/'.join(parts[i]) for test, *_ in staged.values() for i in test}:
            raise ValueError('Mismatch between detection test images and labels')
        stats = {'dataset_sha256': digest, 'threshold_rule': 'maximum validation F1, larger threshold wins ties', 'categories': {}}
        test_records, plot_data = [], {}
        for category, (idx, thresholds, validation, scores) in staged.items():
            y = np.array([int(labels['/'.join(parts[i])]['label'] == 'bad') for i in idx])
            pred = {n: s >= thresholds[n] for n, s in scores.items()}
            measured = {n: classification_metrics(y, scores[n], pred[n]) for n in ANOMALY_RUNS}
            examples = []
            for name in ANOMALY_RUNS:
                for kind, mask in [('false_positive', (y == 0) & pred[name]), ('false_negative', (y == 1) & ~pred[name])]:
                    options = np.flatnonzero(mask)
                    if len(options):
                        relative = scores[name][options] - thresholds[name]
                        picked = options[np.argmax(relative) if kind == 'false_positive' else np.argmin(relative)]
                        examples.append({'method':name, 'error':kind, 'path':paths[idx[picked]], 'score':float(scores[name][picked])})
            stats['categories'][category] = {'fit_counts':[160,40], 'validation_counts':[40,10],
                'test_counts':np.bincount(y,minlength=2).tolist(), 'thresholds':thresholds,
                'validation':validation, 'test':measured, 'error_examples':examples}
            plot_data[category] = (y, scores, examples)
            for k, i in enumerate(idx):
                test_records.append((paths[i], category, int(y[k]), *(v for n in ANOMALY_RUNS for v in (scores[n][k],int(pred[n][k])))))
        stats['macro_mean'] = {n:{metric:float(np.mean([record['test'][n][metric] for record in stats['categories'].values()]))
                                  for metric in ('auroc','ap','precision','recall','f1')} for n in ANOMALY_RUNS}
        write_json(out/'anomaly-metrics.json', stats)
        write_csv(out/'anomaly-predictions.csv', ['image','category','label',*(n+s for n in ANOMALY_RUNS for s in ('_score','_prediction'))], test_records)
        write_csv(out/'anomaly-train-split.csv', ['image','category','partition','training_label'], split_records)
        plot_anomaly(archive, plot_data, out)
    return stats


def plot_anomaly(archive, data, out):
    names = {'global_logistic':'Global LR', 'local_neighbors':'Local NN'}
    colors = {'global_logistic':'#a8551f', 'local_neighbors':'#186ab1'}
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.25), layout='constrained', gridspec_kw={'width_ratios':[1,1,1.2]})
    for row, category in enumerate(CATEGORIES):
        y, scores, examples = data[category]
        for name in ANOMALY_RUNS:
            precision, recall, _ = precision_recall_curve(y, scores[name])
            fpr, tpr, _ = roc_curve(y, scores[name])
            axes[row,0].plot(fpr, tpr, label=names[name], color=colors[name])
            axes[row,1].plot(recall, precision, color=colors[name])
        axes[row,0].plot([0,1], [0,1], '--', color='.55', lw=.8)
        axes[row,1].axhline(y.mean(), ls='--', color='.55', lw=.8)
        axes[row,0].set(title=f'{category.capitalize()} ROC', xlabel='False positive rate', ylabel='True positive rate')
        axes[row,1].set(title=f'{category.capitalize()} PR', xlabel='Recall', ylabel='Precision')
        axes[row,0].legend(loc='lower right', fontsize=7)
        for ax in axes[row,:2]:
            ax.set(xlim=(0,1),ylim=(0,1.02))
            ax.grid(alpha=.2)
        preferred = next((e for e in examples if e['method'] == 'local_neighbors'), None)
        if preferred is None:
            preferred = next((e for e in examples if e['error'] == 'false_negative'), None)
        ax=axes[row,2]
        if preferred:
            ax.imshow(read_image(archive, preferred['path'], (144,144)))
            kind = 'False alarm' if preferred['error']=='false_positive' else 'Missed defect'
            ax.set_title(f"{kind} ({names[preferred['method']]})\n{PurePosixPath(preferred['path']).name}", fontsize=8)
        else:
            ax.text(.5,.5,'No local-NN errors\nat the validation threshold',ha='center',va='center')
        ax.axis('off')
    save_plot(fig, out/'anomaly-results')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cluster','anomaly','all'), default='all')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cache', type=Path, default=Path(__file__).resolve().parent/'.image-cache')
    parser.add_argument('--figure-font', type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(args.cache/'torch-hub'))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.figure_font:
        font_manager.fontManager.addfont(args.figure_font)
        family=font_manager.FontProperties(fname=args.figure_font).get_name()
    else:
        family='STIXGeneral'
    plt.rcParams.update({'font.family':family,'font.size':8,'pdf.fonttype':42})
    started=time.monotonic()
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().cpu()
    with threadpool_limits(limits=1):
        if args.task in ('all','cluster'):
            print('Clustering:', cluster_run(args.data/'Cluster.zip',args.out,model,args.cache)['metrics'])
        if args.task in ('all','anomaly'):
            print('Anomaly detection:', anomaly_run(args.data/'Image_Anomaly_Detection.zip',args.out,model,args.cache)['macro_mean'])
    weight = args.cache/'torch-hub/checkpoints/resnet18-f37072fd.pth'
    with weight.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    write_json(args.out/'image-environment.json', {'torch':str(torch.__version__), 'torchvision':torchvision.__version__,
               'sklearn':sklearn.__version__, 'numpy':np.__version__, 'pillow':PIL.__version__,
               'python':platform.python_version(), 'device':'cpu','weight_sha256':digest,
               'figure_font':family})
    print(f'Completed in {time.monotonic()-started:.1f} seconds')


if __name__=='__main__':
    main()
