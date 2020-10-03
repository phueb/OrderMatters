"""
Compute conditional entropy of distribution of all words in sequence, given test_word word in the same sequence.
This considers left and right and nonadjacent dependencies and is therefore a more complete indicator
of how test_word words are redundant with their neighbors.

"""

import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from pyitlib import discrete_random_variable as drv
import matplotlib.pyplot as plt

from ordermatters import configs
from ordermatters.figs import make_info_theory_fig
from ordermatters.utils import make_prep_from_naturalistic, make_windows, make_test_words

# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191206'
WORDS_NAME = 'sem-4096'
# WORDS_NAME = 'nouns-2972'
DISTANCE = + 1  # don't use negative int, due to bidirectional window
REMOVE_SYMBOLS = None
REMOVE_NUMBERS = True

Y_LIMS = [-0.5, 0.5]


def collect_data(ws: np.ndarray) -> float:

    # test_word windows
    row_ids = np.isin(ws[:, DISTANCE], test_word_ids)
    test_word_windows = ws[row_ids]

    # make multi-variable array with different variables (slots) in rows
    x = test_word_windows.T
    iii = drv.information_interaction(x).item()

    print(f'{len(ws):>12,} | ii={iii:.2f}')

    return iii


# get data
prep = make_prep_from_naturalistic(CORPUS_NAME, REMOVE_SYMBOLS,
                                   context_size=DISTANCE * 2)  # +1 is added by prep to make "window")
test_words = make_test_words(prep, CORPUS_NAME, WORDS_NAME, REMOVE_NUMBERS)
test_words = set(test_words)
test_word_ids = [prep.store.w2id[w] for w in test_words]
windows = make_windows(prep)

# collect results in parallel
pool = Pool(configs.Constants.num_processes)
ii = [[], []]
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
for n, windows in enumerate([windows, np.flip(windows, 0)]):
    print()
    for iii in pool.map(collect_data, [windows[:num_windows] for num_windows in x_ticks]):
        ii[n].append(iii)
pool.close()


# fig
title = f'Cumulative information-interaction between\n' \
        f'{WORDS_NAME} words and words in {DISTANCE * 2}-word bidirectional window\n' \
        f'number words removed={REMOVE_NUMBERS}'
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Entropy [bits]'
labels1 = ['Int(X1, ..., Xn)']
labels2 = ['age-ordered', 'reverse age-ordered']
fig, ax = make_info_theory_fig([ii],
                               title,
                               x_axis_label,
                               y_axis_label,
                               x_ticks,
                               labels1,
                               labels2,
                               y_lims=Y_LIMS,
                               )
plt.show()
