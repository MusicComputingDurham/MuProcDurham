import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from pitchtypes import EnharmonicPitchClass, EnharmonicPitch


def bigram_matrix_from_model(n_gram_model, counts=False):
    min_val = min(n_gram_model.alphabet)
    max_val = max(n_gram_model.alphabet)
    n = max_val - min_val + 1
    if counts:
        mat = np.zeros((n, n), dtype=int)
    else:
        mat = np.zeros((n, n))
    for from_idx, from_symbol in enumerate(range(min_val, max_val + 1)):
        for to_idx, to_symbol in enumerate(range(min_val, max_val + 1)):
            if counts:
                mat[from_idx, to_idx] = n_gram_model.c((from_symbol, to_symbol))
            else:
                mat[from_idx, to_idx] = n_gram_model.p((from_symbol, to_symbol))
    return mat, (min_val, max_val)


def show_bigram_matrix(mat, minmax=None, show_values=False, names=False, pitch_classes=False, origin='upper', show=True,
                       key_depth=3, black_key_depth=2):
    assert mat.shape[0] == mat.shape[1], f"Bigram matrix must be square, but has shape {mat.shape}"
    assert np.all(mat >= 0), "Values in 'mat' must be non-negative"
    if origin not in ['upper', 'lower']:
        raise ValueError(f"'origin' has to be 'upper' or 'lower' but provided value was: '{origin}'")
    black_key_depth = min(black_key_depth, key_depth)
    # get basic heatmap
    norm = plt.Normalize(vmin=0)
    img1 = cm.viridis(norm(mat))
    # add piano keys if minmax is provided
    if minmax is None:
        # minmax not provided: use basic heatmap
        extent = np.array([0, mat.shape[0] - 1, 0, mat.shape[1] - 1]) - 0.5
        value_extent = extent
        img2 = img1
    else:
        # minmax IS provided: add row/column with b/w piano keys
        min_val, max_val = minmax
        img2 = np.ones((img1.shape[0] + key_depth, img1.shape[1] + key_depth, img1.shape[2]), dtype=img1.dtype)
        img2[key_depth:, key_depth:] = img1
        idx = np.arange(img1.shape[0]) + key_depth
        pitch = idx + min_val
        black_keys = (((pitch - 5 - key_depth) % 12) * 7) % 12 > 6
        img2[0:black_key_depth, idx[black_keys], 0:3] = 0  # x-axis
        if origin == 'upper':
            img2[idx[black_keys], key_depth - black_key_depth:key_depth, 0:3] = 0  # y-axis
        else:  # 'lower'
            img2[idx[black_keys], 0:black_key_depth, 0:3] = 0  # y-axis
        value_extent = np.array([min_val, max_val + 1, min_val, max_val + 1]) - 0.5
        extent = value_extent - [key_depth, 0, key_depth, 0]
    plt.imshow(img2, origin='lower', extent=extent)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if origin == 'upper':
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
    if show_values:
        for i, x in enumerate(np.linspace(value_extent[0] + 0.5, value_extent[1] - 0.5, mat.shape[0], endpoint=True)):
            for j, y in enumerate(np.linspace(value_extent[2] + 0.5, value_extent[3] - 0.5, mat.shape[1], endpoint=True)):
                plt.text(x, y, str(mat[j, i]), ha='center', va='center')
    plt.ylabel("from")
    plt.xlabel("to")
    if names:
        if pitch_classes:
            def name_func(x):
                return str(EnharmonicPitchClass(x))
        else:
            def name_func(x):
                return str(EnharmonicPitch(x))
        plt.gca().xaxis.set_major_formatter(lambda x, pos: name_func(x))
        plt.gca().yaxis.set_major_formatter(lambda x, pos: name_func(x))
    plt.colorbar(cm.ScalarMappable(norm=norm), ax=plt.gca())
    if show:
        plt.show()


def show_unigram_distribution(n_gram_model, counts=False, sort=False, show=True, names=False, pitch_classes=False):
    k = len(n_gram_model.alphabet)
    p = np.zeros(k)
    alphabet = n_gram_model.alphabet
    if sort:
        alphabet = sorted(alphabet)
    for i, a in enumerate(alphabet) :
        if counts:
            p[i] = n_gram_model.c((a,))
        else:
            p[i] = n_gram_model.p((a,))
    plt.bar(np.arange(k), p)
    if names:
        if pitch_classes:
            alphabet = [EnharmonicPitchClass(a) for a in alphabet]
        else:
            alphabet = [EnharmonicPitch(a) for a in alphabet]
    plt.xticks(np.arange(k), alphabet)
    if show:
        plt.show()
