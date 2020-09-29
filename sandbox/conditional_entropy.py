"""
Research question:
is the connection between a nouns-next word P less lexically specific (higher conditional entropy) in p1 vs p2?
if so, this would support the idea that nouns are learned more abstractly/flexibly in p1.

conditional entropy (x, y) = how much moe information I need to figure out what X is when Y is known.

so if y is the probability distribution of next-words, and x is P over nouns,
 the hypothesis is that conditional entropy is higher in partition 1 vs. 2

"""

from pyitlib import discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs

from ordermatters import configs
from ordermatters.figs import add_double_legend

# CORPUS_NAME = 'childes-20191206'
CORPUS_NAME = 'newsela'
# WORDS_NAME = 'nouns-2972'
WORDS_NAME = 'numbers'
DISTANCE = -1
NUM_TICKS = 32
NUM_TYPES = 4096 * 4 if CORPUS_NAME == 'newsela' else 4096

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


words_file_path = configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt'
test_words = set([w for w in words_file_path.read_text().split('\n') if w in prep.store.w2id])
print(f'num test_words={len(test_words)}')

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
print(f'Matrix containing all windows has shape={windows.shape}')

# collect data
x_ticks = [int(i) for i in np.linspace(0, len(windows), NUM_TICKS + 1)][1:]
ce1, je1, ye1 = collect_data(windows, reverse=False)
ce2, je2, ye2 = collect_data(windows, reverse=True)

# fig
fig, ax = plt.subplots(1, figsize=(6, 4), dpi=163)
fontsize = 12
plt.title(f'Cumulative uncertainty about {WORDS_NAME} words\n'
          f'given neighbor at distance={DISTANCE}', fontsize=fontsize)
ax.set_ylabel('Entropy [bits]', fontsize=fontsize)
ax.set_xlabel(f'Location in {CORPUS_NAME} [num tokens]', fontsize=fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(x_ticks)
ax.set_xticklabels(['' if n + 1 != len(x_ticks) else i for n, i in enumerate(x_ticks)])
if Y_LIMS:
    ax.set_ylim(Y_LIMS)
# plot conditional entropy
l1, = ax.plot(x_ticks, ce1, '-', linewidth=2, color='C0')
l2, = ax.plot(x_ticks, ce2, '-', linewidth=2, color='C1')
# plot joint entropy
l3, = ax.plot(x_ticks, je1, ':', linewidth=2, color='C0')
l4, = ax.plot(x_ticks, je2, ':', linewidth=2, color='C1')
# plot joint entropy
l5, = ax.plot(x_ticks, ye1, '--', linewidth=2, color='C0')
l6, = ax.plot(x_ticks, ye2, '--', linewidth=2, color='C1')


lines_list = [[l1, l3, l5], [l2, l4, l6]]
labels1 = ['H(X|Y)', 'H(X,Y)', 'H(Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
add_double_legend(lines_list, labels1, labels2)

plt.show()



