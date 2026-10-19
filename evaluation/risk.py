"""Tie-aware classification metrics and paired bootstrap over simulation seeds."""
import numpy as np


def binary_metrics(target, probability, threshold=.5, constant=None):
    y = np.asarray(target, bool); p = np.asarray(probability, float)
    if not len(y):
        return dict(count=0, positives=0, negatives=0, auroc=None, average_precision=None,
                    brier=None, ece=None, prevalence=None, confusion=None)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probabilities')
    order = np.argsort(-p, kind='stable'); ys = y[order]; ps = p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps)-1]
    tp = np.cumsum(ys)[ends]; fp = ends+1-tp
    pos = int(y.sum()); neg = len(y)-pos
    auc = ap = None
    if pos and neg:
        tpr = np.r_[0., tp/pos]; fpr = np.r_[0., fp/neg]
        auc = float(np.trapz(tpr, fpr))
    if pos:
        recall = np.r_[0., tp/pos]
        ap = float(np.sum(np.diff(recall)*tp/(tp+fp)))
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.mean(bins == i)*abs(float(p[bins == i].mean())-
                float(y[bins == i].mean())) for i in range(10) if np.any(bins == i))
    conf = None
    if threshold is not None:
        pred = p >= threshold
        conf = dict(tp=int((pred & y).sum()), fp=int((pred & ~y).sum()),
                    tn=int((~pred & ~y).sum()), fn=int((~pred & y).sum()))
    result = dict(count=len(y), positives=pos, negatives=neg, auroc=auc,
        average_precision=ap, brier=float(np.mean((p-y)**2)), ece=float(ece),
        prevalence=float(y.mean()), threshold=threshold, confusion=conf)
    if constant is not None:
        result['constant_train_prevalence_brier'] = float(np.mean((constant-y)**2))
    return result


def validation_threshold(target, probability):
    y = np.asarray(target, bool); p = np.asarray(probability)
    if not len(y) or y.all() or not y.any():
        return None
    choices = np.unique(np.r_[0., p, 1.])
    best = -np.inf; selected = None
    for threshold in choices:
        pred = p >= threshold
        score = float(pred[y].mean()-pred[~y].mean())
        if score > best + 1e-12:
            best = score; selected = float(threshold)
    return selected


def paired_seed_interval(first, second, seeds, repetitions=2000, seed=20261021):
    first = np.asarray(first); second = np.asarray(second); seeds = np.asarray(seeds)
    if not len(first):
        return dict(delta=None, ci95=None, seed_clusters=0)
    unique = np.unique(seeds); difference = first-second
    sums = np.array([difference[seeds == s].sum() for s in unique])
    counts = np.array([(seeds == s).sum() for s in unique])
    rng = np.random.default_rng(seed)
    index = rng.integers(len(unique), size=(repetitions, len(unique)))
    samples = sums[index].sum(1)/counts[index].sum(1)
    return dict(delta=float(difference.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                seed_clusters=len(unique))
