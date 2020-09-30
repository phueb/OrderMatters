"""
Develop a new quantity for quantifying whether a corpus "starts-good"

= ce1 - ce2, where
ce1 = H(test_word|context)
ce2 = H(test_word vs non-test_word|context)

a "good-start" should correlate with a large value, thus ce11-ce21 > ce12-ce22

# TODO test this quantity on toy corpus

"""

import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from pyitlib import discrete_random_variable as drv
import matplotlib.pyplot as plt

from ordermatters import configs
from ordermatters.figs import make_info_theory_fig
from ordermatters.utils import make_prep, make_windows, make_test_words

# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191206'
WORDS_NAME = 'sem-4096'
REMOVE_NUMBERS = False  # TODO test
DISTANCE = + 1
REMOVE_SYMBOLS = None

Y_LIMS = None


def collect_data(ws: np.ndarray) -> Tuple[float, float]:
    ###############
    # ce1: use only test_word windows
    ###############

    # test_word windows
    row_ids = np.isin(ws[:, -2], test_word_ids)
    test_word_windows = ws[row_ids]

    # ce1
    x1 = test_word_windows[:, -2]  # test_word
    y1 = test_word_windows[:, -2 + DISTANCE]  # neighbors
    ce1i = drv.entropy_conditional(x1, y1).item()

    ###############
    # ce2: use all windows, and make observations in x categorical (test_word vs. not-a-test_word)
    ###############

    # ce2
    x2 = ws[:, -2]  # all words
    y2 = ws[:, -2 + DISTANCE]  # neighbors
    x2 = [1 if prep.store.types[i] in test_words else 0 for i in x2]  # make binary
    ce2i = drv.entropy_conditional(x2, y2).item()

    print(f'{len(ws):>12,} | ce1={ce1i:.2f} ce2={ce2i:.2f}')

    return ce1i, ce2i


# get data
prep = make_prep(CORPUS_NAME, REMOVE_SYMBOLS)
test_words = make_test_words(prep, CORPUS_NAME, WORDS_NAME, REMOVE_NUMBERS)
test_words = set(test_words)
test_word_ids = [prep.store.w2id[w] for w in test_words]
windows = make_windows(prep)

# collect data in parallel
pool = Pool(configs.Constants.num_processes)
ce1 = [[], []]
ce2 = [[], []]
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
for n, windows in enumerate([windows, np.flip(windows, 0)]):
    print()
    for ce1i, ce2i in pool.map(collect_data, [windows[:num_windows] for num_windows in x_ticks]):
        ce1[n].append(ce1i)
        ce2[n].append(ce2i)
pool.close()

# fig
title = f'"Cumulative info about items vs. category" about {WORDS_NAME} words\n' \
        f'given neighbor at distance={DISTANCE}'
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Entropy [bits]'
labels1 = ['H(X|Y)', 'H(categorical(X)|Y)']  # , 'H(X|Y) - H(categorical(X)|Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
fig, ax = make_info_theory_fig([ce1,
                                ce2,
                                # [np.subtract(ce11, ce21), np.subtract(ce12, ce22)]
                                ],
                               title,
                               x_axis_label,
                               y_axis_label,
                               x_ticks,
                               labels1,
                               labels2,
                               y_lims=Y_LIMS,
                               )
plt.show()
