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
from ordermatters.figs import make_info_theory_fig

CORPUS_NAME = 'newsela'
# CORPUS_NAME = 'childes-20191206'
# WORDS_NAME = 'sem-4096'
WORDS_NAME = 'numbers'
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
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
print(f'Matrix containing all windows has shape={windows.shape}')

# collect data
mi1 = collect_data(windows, reverse=False)
mi2 = collect_data(windows, reverse=True)

# fig
title = f'x={DISTANCE}\ny={WORDS_NAME}\nnorm={NORM_FACTOR}\n' \
        f'(Nouns are NOT binary outcomes)'
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Normalized Mutual Info'
labels1 = ['I(X;Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
fig, ax = make_info_theory_fig([[mi1, mi2]],
                               title,
                               x_axis_label,
                               y_axis_label,
                               x_ticks,
                               labels1,
                               labels2,
                               y_lims=Y_LIMS,
                               )
plt.show()
