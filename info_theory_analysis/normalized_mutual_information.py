"""
Research question:
Is mutual information between a probe and a neighbor higher in partition 1 of AO-CHILDES?

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
DISTANCE = + 1
REMOVE_SYMBOLS = None
REMOVE_NUMBERS = True

NORM_FACTOR = 'XY'

Y_LIMS = None


def collect_data(ws: np.ndarray) -> float:

    # probe windows
    row_ids = np.isin(ws[:, -2], test_word_ids)
    probe_windows = ws[row_ids]

    # mutual info
    assert DISTANCE <= 1
    x = probe_windows[:, -2]  # probe
    y = probe_windows[:, -2 + DISTANCE]  # left or right context
    mii = drv.information_mutual_normalised(x, y, norm_factor=NORM_FACTOR)

    print(f'{len(ws):>12,} | mi={mii:.2f}')

    return mii


# get data
prep = make_prep(CORPUS_NAME, REMOVE_SYMBOLS)
test_words = make_test_words(prep, CORPUS_NAME, WORDS_NAME, REMOVE_NUMBERS)
test_words = set(test_words)
test_word_ids = [prep.store.w2id[w] for w in test_words]
windows = make_windows(prep)

# collect results in parallel
pool = Pool(configs.Constants.num_processes)
mi = [[], []]
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
for n, windows in enumerate([windows, np.flip(windows, 0)]):
    print()
    for mii in pool.map(collect_data, [windows[:num_windows] for num_windows in x_ticks]):
        mi[n].append(mii)
pool.close()


# fig
title = f'x={DISTANCE}\ny={WORDS_NAME}\nnorm={NORM_FACTOR}\n' \
        f'number words removed={REMOVE_NUMBERS}'
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Normalized Mutual Info'
labels1 = ['I(X;Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
fig, ax = make_info_theory_fig([mi],
                               title,
                               x_axis_label,
                               y_axis_label,
                               x_ticks,
                               labels1,
                               labels2,
                               y_lims=Y_LIMS,
                               )
plt.show()
