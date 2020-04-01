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
from ordermatters.figs import add_double_legend

CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'sem-4096'
NUM_TICKS = 32
NUM_TYPES = 4096
DISTANCE = 4

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

num_windows_list = [int(i) for i in np.linspace(0, len(windows), NUM_TICKS + 1)][1:]


def collect_data(windows, reverse: bool):

    if reverse:
        windows = np.flip(windows, 0)

    ce = []
    je = []
    for num_windows in num_windows_list:

        ws = windows[:num_windows]
        print(num_windows, ws.shape)

        # probe windows
        row_ids = np.isin(ws[:, -2], [prep.store.w2id[w] for w in probes])
        probe_windows = ws[row_ids]
        print(f'num probe windows={len(probe_windows)}')

        x = probe_windows[:, -(2 + DISTANCE)]  # left context
        y = probe_windows[:, -2]  # CAT member

        cei = drv.entropy_conditional(y, x)  # order here is important: y, x

        x_y = np.vstack((x, y))
        jei = drv.entropy_joint(x_y)

        print(f'cei={cei}')
        print(f'jei={jei}')
        print()
        ce.append(cei)
        je.append(jei)

    return ce, je


# collect data
ce1, je1 = collect_data(windows, reverse=False)
ce2, je2 = collect_data(windows, reverse=True)

# fig
fig, ax = plt.subplots(1, figsize=(6, 5), dpi=163)
fontsize = 12
plt.title(f'Cumulative uncertainty about {PROBES_NAME} given left-word'
          f'\n(Nouns are NOT binary outcomes)'
          f'\ndistance={DISTANCE}', fontsize=fontsize)
ax.set_ylabel('Entropy [bits]', fontsize=fontsize)
ax.set_xlabel('Location in AO-CHILDES [num tokens]', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(num_windows_list)
ax.set_xticklabels([i if n in [0, len(num_windows_list) - 1] else '' for n, i in enumerate(num_windows_list)])
# ax.set_ylim([7.0, 8.0])
# plot conditional entropy
l1, = ax.plot(num_windows_list, ce1, '-', linewidth=2, color='C0')
l2, = ax.plot(num_windows_list, ce2, '-', linewidth=2, color='C1')
# plot joint entropy
l3, = ax.plot(num_windows_list, je1, ':', linewidth=2, color='C0')
l4, = ax.plot(num_windows_list, je2, ':', linewidth=2, color='C1')

lines_list = [[l1, l3], [l2, l4]]
labels1 = ['conditional', 'joint']
labels2 = ['age-ordered', 'reverse age-ordered']
add_double_legend(lines_list, labels1, labels2)

plt.show()



