"""
A new quantity for quantifying whether a corpus "starts-good":

mi1 - mi2, where
mi1 measures the relationship between a category (defined by test words) and their neighbors
mi2 measures the relationship between individual test words and their neighbors

a "good start should have large mi1 and small mi2


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


TOY_DATA = True

# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191206'
WORDS_NAME = 'sem-4096'
DISTANCE = + 1
REMOVE_NUMBERS = True
REMOVE_SYMBOLS = None

Y_LIMS = [-0.5, 0.0]


def collect_data(ws: np.ndarray) -> Tuple[float, float]:

    ###############
    # val1: use all windows, and make observations in x categorical (test_word vs. not-a-test_word)
    ###############

    # val1
    x1 = ws[:, -2]  # all words
    y1 = ws[:, -2 + DISTANCE]  # neighbors
    x1 = [1 if prep.store.types[i] in test_words else 0 for i in x1]  # make categorical
    val1i = drv.information_mutual_normalised(x1, y1).item()

    ###############
    # val2: use only test_word windows
    ###############

    # test_word windows
    row_ids = np.isin(ws[:, -2], test_word_ids)
    test_word_windows = ws[row_ids]

    # val2
    x2 = test_word_windows[:, -2]  # test_word
    y2 = test_word_windows[:, -2 + DISTANCE]  # neighbors
    val2i = drv.information_mutual_normalised(x2, y2).item()

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
    test_words = make_test_words(prep, CORPUS_NAME, WORDS_NAME, REMOVE_NUMBERS)

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

# fig - val 1
title = f'Component 1\n' + title_additional
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Mutual Information [bits]'
labels1 = ['I(categorical(X);Y)']
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
)
plt.show()

# fig - val 2
title = f'Component 2\n'  + title_additional
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Mutual Information [bits]'
labels1 = ['I(X;Y)']
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
)
plt.show()

# fig - both values together
title = f'Cumulative info about test word category - items\n' + title_additional
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Mutual Information [bits]'
labels1 = ['I(categorical(X);Y) - I(X;Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
make_info_theory_fig([
    np.subtract(val1, val2),
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
