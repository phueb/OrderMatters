"""
A new quantity for quantifying whether a corpus "starts-good":

ce_xy2 - ce_xy1, where
ce_xy1 measures the uncertainty about all words in corpus given their neighbors
ce_xy2 measures the uncertainty about individual test words given their neighbors

a "good start should have large difference (e.g high ce_xy2, and low ce_xy1)


"""

import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from pyitlib import discrete_random_variable as drv
import matplotlib.pyplot as plt

from preppy import PartitionedPrep

from ordermatters.corpus_toy import ToyCorpus
from ordermatters import configs
from ordermatters.figs import make_info_theory_fig
from ordermatters.utils import make_prep_from_naturalistic, make_windows, make_test_words


TOY_DATA = False

# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191206'
WORDS_NAME = 'childes-20191206-sem-4096'
DISTANCE = + 1
REMOVE_NUMBERS = True
REMOVE_SYMBOLS = None

SHOW_COMPONENTS = True


def collect_data(ws: np.ndarray) -> Tuple[float, float]:

    ###############
    # val1: use all windows; this provides a control/reference from which to measure difference to val2
    # (should be 0.0 when using toy corpus)
    ###############

    # val1
    x1 = ws[:, -2]  # all words
    y1 = ws[:, -2 + DISTANCE]  # neighbors
    val1i = drv.entropy_conditional(x1, y1).item() / drv.entropy(x1).item()

    ###############
    # val2: use only test_word windows
    # in theory, this should be invariant to number of test-word types,
    ###############

    # test_word windows
    row_ids = np.isin(ws[:, -2], test_word_ids)
    test_word_windows = ws[row_ids]

    # val2
    x2 = test_word_windows[:, -2]  # test_word
    y2 = test_word_windows[:, -2 + DISTANCE]  # neighbors
    val2i = drv.entropy_conditional(x2, y2).item() / drv.entropy(x2).item()

    print(f'{len(ws):>12,} | val1={val1i:.3f} val2={val2i:.2f}')

    return val1i, val2i


# get data from toy corpus
if TOY_DATA:
    tc = ToyCorpus()
    prep = PartitionedPrep(tc.docs,
                           reverse=False,
                           num_types=tc.num_types,
                           num_parts=2,
                           num_iterations=(1, 1),
                           batch_size=64,
                           context_size=1)
    test_words = [p for p in tc.nouns if p in prep.store.w2id]

# get data from naturalistic corpus
else:
    tc = None
    prep = make_prep_from_naturalistic(CORPUS_NAME, REMOVE_SYMBOLS)
    test_words = make_test_words(prep, WORDS_NAME, REMOVE_NUMBERS)

test_words = set(test_words)
test_word_ids = [prep.store.w2id[w] for w in test_words]
windows = make_windows(prep)

# collect results in parallel
pool = Pool(configs.Constants.num_processes)
val1 = [[], []]
val2 = [[], []]
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
for n, windows in enumerate([windows, np.flip(windows, 0)]):
    print()
    for val1i, val2i in pool.map(collect_data, [windows[:num_windows] for num_windows in x_ticks]):
        val1[n].append(val1i)
        val2[n].append(val2i)
pool.close()

if TOY_DATA:
    title_additional = f'toy data with num_docs={tc.num_docs} and doc_size={tc.doc_size}\n'\
                       f'increase_noun_types={tc.increase_noun_types}\n' \
                       f'increase_other_types={tc.increase_other_types}\n'
else:
    title_additional = f'test words={WORDS_NAME}\n' \
                       f'neighbors at distance={DISTANCE}\n' \
                       f'remove number words={REMOVE_NUMBERS}\n'

if SHOW_COMPONENTS:
    # fig - val 1
    title = f'Component 1\n' + title_additional
    x_axis_label = f'Location in {"toy data" if TOY_DATA else CORPUS_NAME} [num tokens]'
    y_axis_label = 'Normalized Entropy'
    labels1 = ['H(all X|Y) / H(all X)']
    labels2 = ['age-ordered', 'reverse age-ordered']
    make_info_theory_fig([
        val1,
    ],
        title,
        x_axis_label,
        y_axis_label,
        x_ticks,
        labels1,
        labels2,
        y_lims=[0, 1],
    )
    plt.show()

    # fig - val 2
    title = f'Component 2\n' + title_additional
    x_axis_label = f'Location in {"toy data" if TOY_DATA else CORPUS_NAME} [num tokens]'
    y_axis_label = 'Normalized Entropy'
    labels1 = ['H(test-word X|Y) / H(test-word X)']
    labels2 = ['age-ordered', 'reverse age-ordered']
    make_info_theory_fig([
        val2,
    ],
        title,
        x_axis_label,
        y_axis_label,
        x_ticks,
        labels1,
        labels2,
        y_lims=[0, 1],
    )
    plt.show()

# fig - both values together
title = f'Component 1 - Component 2\n' + title_additional
x_axis_label = f'Location in {"toy data" if TOY_DATA else CORPUS_NAME} [num tokens]'
y_axis_label = 'Normalized Entropy'
labels1 = ['H(test-word X|Y) / H(test-word X) - H(all X|Y) / H(all X)']
labels2 = ['age-ordered', 'reverse age-ordered']
make_info_theory_fig([
    np.subtract(val2, val1),
],
    title,
    x_axis_label,
    y_axis_label,
    x_ticks,
    labels1,
    labels2,
    y_lims=[0, 0.2],
)
plt.show()

if TOY_DATA:
    print('Showing results for toy corpus, not naturalistic corpus')