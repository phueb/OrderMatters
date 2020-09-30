"""
Research question:
is the connection between a nouns-next word P less lexically specific (higher conditional entropy) in p1 vs p2?
if so, this would support the idea that nouns are learned more abstractly/flexibly in p1.

conditional entropy (x, y) = how much moe information I need to figure out what X is when Y is known.

so if y is the probability distribution of next-words, and x is P over nouns,
 the hypothesis is that conditional entropy is higher in partition 1 vs. 2

WARNING: because no normalization is used, it may not be a good idea to directly compare
age-ordered vs reverse age-ordered results,
rendering this script useless > use normalized MI instead

"""

import numpy as np
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

Y_LIMS = None


def collect_data(windows, reverse: bool):

    if reverse:
        windows = np.flip(windows, 0)

    ce = []
    je = []
    ye = []
    for num_windows in x_ticks:

        ws = windows[:num_windows]
        print(f'{num_windows:>12,}/{x_ticks[-1]:>12,}')

        # probe windows
        row_ids = np.isin(ws[:, -2], test_word_ids)
        probe_windows = ws[row_ids]

        # conditional entropy, joint entropy, y entropy
        x = probe_windows[:, -2]  # probe
        y = probe_windows[:, -2 + DISTANCE]  # left or right context
        cei = drv.entropy_conditional(x, y)
        x_y = np.vstack((x, y))
        jei = drv.entropy_joint(x_y)
        yei = drv.entropy(y)  # Y entropy

        ce.append(cei)
        je.append(jei)
        ye.append(yei)

    return ce, je, ye


# get data
prep = make_prep(CORPUS_NAME, REMOVE_SYMBOLS)
test_words = make_test_words(prep, CORPUS_NAME, WORDS_NAME)
test_words = set(test_words)
test_word_ids = [prep.store.w2id[w] for w in test_words]
windows = make_windows(prep)

# collect data
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
ce1, je1, ye1 = collect_data(windows, reverse=False)
ce2, je2, ye2 = collect_data(windows, reverse=True)

# fig
title = f'Cumulative uncertainty about {WORDS_NAME} words\n' \
        f'given neighbor at distance={DISTANCE}'
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Entropy [bits]'
labels1 = ['H(X|Y)', 'H(X,Y)', 'H(Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
fig, ax = make_info_theory_fig([[ce1, ce2], [je1, je2], [ye1, ye2]],
                               title,
                               x_axis_label,
                               y_axis_label,
                               x_ticks,
                               labels1,
                               labels2,
                               y_lims=Y_LIMS,
                               )
plt.show()
