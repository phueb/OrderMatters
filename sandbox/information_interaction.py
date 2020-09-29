"""
Compute conditional entropy of distribution of all words in sequence, given probe word in the same sequence.
This considers left and right and nonadjacent dependencies and is therefore a more complete indicator
of how probe words are redundant with their neighbors.

"""


from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv

from preppy import PartitionedPrep
from preppy.docs import load_docs


from ordermatters.figs import add_double_legend
from ordermatters import configs

CORPUS_NAME = 'newsela'
# CORPUS_NAME = 'childes-20191206'
# WORDS_NAME = 'sem-4096'
WORDS_NAME = 'numbers'
NUM_TICKS = 16
NUM_TYPES = 4096 * 4 if CORPUS_NAME == 'newsela' else 4096
REMOVE_SYMBOLS = None
DISTANCE = 2  # half of bidirectional window, excluding probe

Y_LIMS = [0.5, 1.6]  # [0.0, 3.0]


def collect_data(windows_mat, reverse: bool):

    if reverse:
        windows_mat = np.flip(windows_mat, 0)

    ii = []
    for num_windows in x_ticks:

        ws = windows_mat[:num_windows]
        print(f'{num_windows:>12,}/{x_ticks[-1]:>12,}')

        # probe windows
        row_ids = np.isin(ws[:, DISTANCE], test_word_ids)
        probe_windows = ws[row_ids]

        # make multi-variable array with different variables (slots) in rows
        x = probe_windows.T
        iii = drv.information_interaction(x)

        ii.append(iii)

    return ii


corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path,
                          remove_symbols=REMOVE_SYMBOLS)

prep = PartitionedPrep(train_docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=2,
                       num_iterations=(20, 20),
                       batch_size=64,
                       context_size=DISTANCE * 2,  # +1 is added by prep to make "window"
                       )

test_words = (configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt').open().read().split("\n")
test_word_ids = [prep.store.w2id[w] for w in test_words if w in prep.store.w2id]  # must not be  a set
print(f'Including {len(test_word_ids)} out of {len(test_words)} test_words in file')

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
x_ticks = [int(i) for i in np.linspace(0, len(windows), NUM_TICKS + 1)][1:]
print(f'prep.num_tokens_in_window={prep.num_tokens_in_window}')
print(f'Matrix containing all windows has shape={windows.shape}')

# collect data
ii1 = collect_data(windows, reverse=False)
ii2 = collect_data(windows, reverse=True)

# fig
fig, ax = plt.subplots(1, figsize=(6, 4), dpi=163)
fontsize = 14
plt.title(f'Cumulative information-interaction between {WORDS_NAME} words\n'
          f'and words in {DISTANCE * 2}-word bidirectional window\n', fontsize=fontsize)
ax.set_ylabel('Entropy [bits]', fontsize=fontsize)
ax.set_xlabel(f'Location in {CORPUS_NAME} [num tokens]', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(x_ticks)
ax.set_xticklabels(['' if n + 1 != len(x_ticks) else i for n, i in enumerate(x_ticks)])
if Y_LIMS:
    ax.set_ylim(Y_LIMS)
# plot
l1, = ax.plot(x_ticks, ii1, '-', linewidth=2, color='C0')
l2, = ax.plot(x_ticks, ii2, '-', linewidth=2, color='C1')
# legend
lines_list = [[l1], [l2]]
labels1 = ['Int(X1, ..., Xn)']
labels2 = ['age-ordered', 'reverse age-ordered']
add_double_legend(lines_list, labels1, labels2)
plt.show()
