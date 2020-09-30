"""
Develop a new quantity for quantifying whether a corpus "starts-good"

= ratio of E1 and E2, where
ce1 = H(test_word|context)
ce2 = H(test_word vs non-test_word|context)

a "good-start" should correlate with a large ratio, thus ce11/ce21 > ce12/ce22

# TODO test this quantity on toy corpus

"""

from numpy.lib.stride_tricks import as_strided
import numpy as np
from pyitlib import discrete_random_variable as drv
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs


from ordermatters import configs
from ordermatters.figs import make_info_theory_fig

CORPUS_NAME = 'newsela'
# CORPUS_NAME = 'childes-20191206'
WORDS_NAME = 'sem-4096'
DISTANCE = + 1
NUM_TYPES = 4096 * 4 if CORPUS_NAME == 'newsela' else 4096
REMOVE_SYMBOLS = None

Y_LIMS = None


def collect_data(windows, reverse: bool):

    if reverse:
        windows = np.flip(windows, 0)

    ce1 = []
    ce2 = []
    for num_windows in x_ticks:

        ws = windows[:num_windows]
        print(f'{num_windows:>12,}/{x_ticks[-1]:>12,}')

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

        ce1.append(ce1i)
        ce2.append(ce2i)

    return ce1, ce2


corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path,
                          remove_symbols=REMOVE_SYMBOLS)

prep = PartitionedPrep(train_docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=2,
                       num_iterations=(20, 20),
                       batch_size=64,
                       context_size=7,
                       )

test_words = (configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt').open().read().split("\n")
test_word_ids = [prep.store.w2id[w] for w in test_words if w in prep.store.w2id]
print(f'Including {len(test_word_ids)} out of {len(test_words)} test_words in file')

test_words = set(test_words)

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
print(f'Matrix containing all windows has shape={windows.shape}')

# collect data
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
ce11, ce21 = collect_data(windows, reverse=False)
ce12, ce22 = collect_data(windows, reverse=True)

# fig
title = f'"Info-Ratio" about {WORDS_NAME} words\n' \
        f'given neighbor at distance={DISTANCE}'
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Entropy Ratio [bits]'
labels1 = ['H(X|Y) / H(categorical(X)|Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
fig, ax = make_info_theory_fig([[np.divide(ce11, ce21), np.divide(ce12, ce22)]],
                               title,
                               x_axis_label,
                               y_axis_label,
                               x_ticks,
                               labels1,
                               labels2,
                               y_lims=Y_LIMS,
                               )
plt.show()
