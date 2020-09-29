"""
Research question:
Is mutual information between a probe and a neighbor higher in partition 1 of AO-CHILDES?

"""

from pyitlib import discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs


from ordermatters import configs

CORPUS_NAME = 'newsela'
# CORPUS_NAME = 'childes-20191206'
# WORDS_NAME = 'sem-4096'
WORDS_NAME = 'numbers'
NUM_TICKS = 32
NUM_TYPES = 4096 * 4 if CORPUS_NAME == 'newsela' else 4096
DISTANCE = -1  # can be negative or positive
NORM_FACTOR = 'XY'

Y_LIMS = [0.10, 0.35]


def collect_data(windows_mat, reverse: bool):

    if reverse:
        windows_mat = np.flip(windows_mat, 0)

    res = []
    for num_windows in x_ticks:

        ws = windows_mat[:num_windows]
        print(f'{num_windows:>12,}/{x_ticks[-1]:>12,}')

        # probe windows
        row_ids = np.isin(ws[:, -2], test_word_ids)
        probe_windows = ws[row_ids]

        # mutual info
        assert DISTANCE <= 1
        x = probe_windows[:, -2]  # probe
        y = probe_windows[:, -2 + DISTANCE]  # left or right context
        mi = drv.information_mutual_normalised(x, y, norm_factor=NORM_FACTOR)

        res.append(mi)

    return res


corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = PartitionedPrep(train_docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=2,
                       num_iterations=(20, 20),
                       batch_size=64,
                       context_size=7,
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
print(f'Matrix containing all windows has shape={windows.shape}')

# collect data
mi1 = collect_data(windows, reverse=False)
mi2 = collect_data(windows, reverse=True)

# fig
fig, ax = plt.subplots(1, figsize=(6, 5), dpi=163)
fontsize = 12
plt.title(f'x={DISTANCE}\ny={WORDS_NAME}\nnorm={NORM_FACTOR}'
          f'\n(Nouns are NOT binary outcomes)',
          fontsize=fontsize)
ax.set_ylabel('Normalized Mutual Info', fontsize=fontsize)
ax.set_xlabel(f'Location in {CORPUS_NAME} [num tokens]', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(x_ticks)
ax.set_xticklabels([i if n in [0, len(x_ticks) - 1] else '' for n, i in enumerate(x_ticks)])
if Y_LIMS:
    ax.set_ylim(Y_LIMS)
# plot
ax.plot(x_ticks, mi1, '-', linewidth=2, color='C0', label='age ordered')
ax.plot(x_ticks, mi2, '-', linewidth=2, color='C1', label='reverse age ordered')
plt.legend(frameon=False)
plt.show()
