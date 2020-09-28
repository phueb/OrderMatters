"""
Research question:
Given that it is known whether a word is a noun or not (picking from binary distribution),
how much uncertainty remains about the previous word?

the uncertainty should be lower in partition 1 of AO-CHILDES

"""

from pyitlib import discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs
from categoryeval.probestore import ProbeStore

from ordermatters import configs
from ordermatters.figs import add_double_legend

CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'sem-4096'
NUM_TICKS = 4
NUM_TYPES = 4096

corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = PartitionedPrep(train_docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=2,
                       num_iterations=[20, 20],
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
print(num_windows_list)


def collect_data(windows, reverse: bool):

    if reverse:
        windows = np.flip(windows, 0)

    ce = []
    je = []
    for num_windows in num_windows_list:

        ws = windows[:num_windows]
        print(num_windows, ws.shape)

        x = ws[:, -3]  # last-context-word
        y = ws[:, -2]  # CAT member

        # make observations in w2 binary
        y = [1 if prep.store.types[i] in probes else 0 for i in y]

        cei = drv.entropy_conditional(x, y)

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
fig, ax = plt.subplots(1, figsize=(6, 4), dpi=163)
fontsize = 14
plt.title(f'Cumulative uncertainty about words left-adjacent to {POS}s', fontsize=fontsize)
ax.set_ylabel('Entropy [bits]', fontsize=fontsize)
ax.set_xlabel('Location in AO-CHILDES [num tokens]', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(num_windows_list)
ax.set_xticklabels(num_windows_list)
ax.set_ylim([7.0, 8.0])
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



