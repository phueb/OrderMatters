"""
Research question:
Given that the particular outcome of the noun is known in noun windows,
how much uncertainty remains in predicting a noun given its left context?

the uncertainty should be HIGHER in partition 1 of AO-CHILDES

"""

from pyitlib import discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs
from categoryeval.probestore import ProbeStore

from ordermatters import config

CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'sem-4096'
NUM_TICKS = 32
NUM_TYPES = 4096
DISTANCE = -2  # can be negative or positive
NORM_FACTOR = 'XY'


def collect_data(windows_mat, reverse: bool):

    if reverse:
        windows_mat = np.flip(windows_mat, 0)

    res = []
    for num_windows in x_ticks:

        ws = windows_mat[:num_windows]
        print(f'{num_windows:>12,}/{x_ticks[-1]:>12,}')

        # probe windows
        row_ids = np.isin(ws[:, -2], [prep.store.w2id[w] for w in probes])
        probe_windows = ws[row_ids]

        # mutual info
        assert DISTANCE <= 1
        x = probe_windows[:, -2 + DISTANCE]  # left or right context
        y = probe_windows[:, -2]  # probe
        mi = drv.information_mutual_normalised(x, y, norm_factor=NORM_FACTOR)

        res.append(mi)

    return res


corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = PartitionedPrep(train_docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=2,
                       num_iterations=(20, 20),
                       batch_size=64,
                       context_size=7,
                       )

store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)
probes = store.types
print(f'num probes={len(probes)}')

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
print(f'Matrix containing all windows has shape={windows.shape}')

x_ticks = [int(i) for i in np.linspace(0, len(windows), NUM_TICKS + 1)][1:]


# collect data
mi1 = collect_data(windows, reverse=False)
mi2 = collect_data(windows, reverse=True)

# fig
fig, ax = plt.subplots(1, figsize=(6, 5), dpi=163)
fontsize = 12
plt.title(f'x={DISTANCE}\ny={PROBES_NAME}\nnorm={NORM_FACTOR}'
          f'\n(Nouns are NOT binary outcomes)',
          fontsize=fontsize)
ax.set_ylabel('Cumulative Normalized Mutual Info', fontsize=fontsize)
ax.set_xlabel('Location in AO-CHILDES [num tokens]', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(x_ticks)
ax.set_xticklabels([i if n in [0, len(x_ticks) - 1] else '' for n, i in enumerate(x_ticks)])
# plot
ax.plot(x_ticks, mi1, '-', linewidth=2, color='C0', label='age ordered')
ax.plot(x_ticks, mi2, '-', linewidth=2, color='C1', label='reverse age ordered')
plt.legend(frameon=False)
plt.show()



