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
from ordermatters.figs import make_info_theory_fig

# CORPUS_NAME = 'childes-20191206'
CORPUS_NAME = 'newsela'
# WORDS_NAME = 'nouns-2972'
WORDS_NAME = 'numbers'
DISTANCE = -1
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


# load test words
test_words = (configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt').open().read().split("\n")
test_word_ids = [prep.store.w2id[w] for w in test_words if w in prep.store.w2id]  # must not be  a set
print(f'Including {len(test_word_ids)} out of {len(test_words)} test_words in file')

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
print(f'Matrix containing all windows has shape={windows.shape}')

# collect data
x_ticks = [int(i) for i in np.linspace(0, len(windows), configs.Constants.num_ticks + 1)][1:]
ce1, je1, ye1 = collect_data(windows, reverse=False)
ce2, je2, ye2 = collect_data(windows, reverse=True)

# fig
title = f'Cumulative uncertainty about {WORDS_NAME} words\n' \
        f'given neighbor at distance={DISTANCE}'
x_axis_label = f'Location in {CORPUS_NAME} [num tokens]'
y_axis_label = 'Entropy [bits]'
labels1 = ['H(X|Y)', 'H(X,Y)', 'H(Y)']
labels2 = ['age-ordered', 'reverse age-ordered']
fig, ax = make_info_theory_fig([[ce1, ce2], [je1, je2], [ye1, ye2]],
                               title,
                               x_axis_label,
                               y_axis_label,
                               x_ticks,
                               labels1,
                               labels2,
                               y_lims=Y_LIMS,
                               )
plt.show()



