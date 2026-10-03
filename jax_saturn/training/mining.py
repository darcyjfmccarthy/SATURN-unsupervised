"""CPU cross-species MNN candidate sampling preserving the active reference miner."""

import numpy as np


def normalize_numpy(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def cross_species_candidates(embeddings, labels, species_codes, *, rng=None, mnn=True, randint=None):
    """Sample positives with replacement and negatives from either pair species.

    `randint(low, high, shape)` may be injected for deterministic reference tests.
    Duplicate MNN matches are retained. Degenerate groups without cross-species
    positives or an eligible negative yield no triplets instead of a sampling error.
    """
    values = normalize_numpy(embeddings)
    labels, species = np.asarray(labels), np.asarray(species_codes)
    if values.ndim != 2 or labels.shape != (len(values),) or species.shape != labels.shape:
        raise ValueError("Mining requires matching embedding, label and species rows")
    if not np.isfinite(values).all():
        raise ValueError("Mining embeddings must be finite")
    rng = np.random.default_rng() if rng is None else rng
    choose = rng.integers if randint is None else randint
    output = [[], [], []]
    unique_species = np.unique(species)
    if len(unique_species) < 2:
        return tuple(np.empty(0, dtype=np.int64) for _ in range(3))
    similarity = values @ values.T
    for label in np.unique(labels):
        positives = np.flatnonzero(labels == label)
        positive_species = np.unique(species[positives])
        if len(positive_species) == 1:
            current = positive_species[0]
            forward = similarity[:, positives].copy()
            forward[species == current] = -1
            closest = forward.argmax(axis=0)
            if mnn:
                backward = similarity[:, closest].copy()
                backward[species != current] = -1
                closest = closest[labels[backward.argmax(axis=0)] == label]
            positives = np.concatenate((positives, closest))
        negatives = np.concatenate([np.flatnonzero((labels != label) & (species == code))
                                    for code in unique_species])
        if len(positives) < 2 or not len(negatives):
            continue
        anchors, partners = [], []
        for code in unique_species:
            local = positives[species[positives] == code]
            other = positives[species[positives] != code]
            if len(other):
                size = len(local) * len(other)
                anchors.append(np.repeat(local, len(other)))
                partners.append(other[np.asarray(choose(0, len(other), (size,)))])
        if not anchors:
            continue
        anchors, partners = np.concatenate(anchors), np.concatenate(partners)
        sampled_negative = np.zeros(len(anchors), dtype=np.int64)
        usable = np.ones(len(anchors), dtype=bool)
        for anchor_label in np.unique(labels[anchors]):
            for positive_label in np.unique(labels[partners]):
                a_species = species[labels == anchor_label][0]
                p_species = species[labels == positive_label][0]
                pool = negatives[(labels[negatives] != anchor_label) & (labels[negatives] != positive_label)
                                 & ((species[negatives] == a_species) | (species[negatives] == p_species))]
                rows = np.flatnonzero((labels[anchors] == anchor_label) & (labels[partners] == positive_label))
                if not len(rows):
                    continue
                if not len(pool):
                    usable[rows] = False
                    continue
                sampled_negative[rows] = pool[np.asarray(choose(0, len(pool), (len(rows),)))]
        for destination, indices in zip(output, (anchors, partners, sampled_negative)):
            destination.append(indices[usable])
    return tuple(np.concatenate(items).astype(np.int64) if items else np.empty(0, dtype=np.int64)
                 for items in output)


def filter_triplets(embeddings, indices, *, margin=.2, kind="semihard"):
    values = normalize_numpy(embeddings)
    a, p, n = indices
    # Separate dot products match the reference miner's similarity matrix entries.
    difference = np.sum(values[a] * values[p], axis=-1) - np.sum(values[a] * values[n], axis=-1)
    if kind == "semihard":
        mask = (difference > 0) & (difference <= margin)
    elif kind == "hard":
        mask = (difference <= 0) & (difference <= margin)
    elif kind == "all":
        mask = difference <= margin
    elif kind == "unfiltered":
        mask = np.ones(len(a), dtype=bool)
    else:
        raise ValueError("Unsupported baseline triplet filter")
    return tuple(item[mask] for item in indices)


def pad_triplets(indices, *, minimum=256):
    """Bucket dynamic tuples by powers of two; never truncate scientific candidates."""
    count = len(indices[0])
    if any(len(item) != count for item in indices):
        raise ValueError("Triplet tuple lengths differ")
    capacity = max(minimum, 1 << max(count - 1, 0).bit_length())
    padded = tuple(np.pad(np.asarray(item, dtype=np.int32), (0, capacity - count)) for item in indices)
    return padded, np.arange(capacity) < count
