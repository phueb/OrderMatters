from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs
from categoryeval.probestore import ProbeStore

from ordermatters import configs
from ordermatters.figs import make_example_fig

CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'sem-4096'
NUM_TYPES = 4096 * 4 if CORPUS_NAME == 'newsela' else 4096
REMOVE_SYMBOLS = None
NUM_ROWS = 512

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

store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)
probes = store.types

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

# probe windows
row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
probe_windows = windows[row_ids]

x_actual = probe_windows[:, -2]  # CAT member
y_actual = probe_windows[:, -1]  # next-word

# map word ID of nouns to IDs between [0, len(probes)]
# this makes creating a matrix with the right number of columns easier
x2x = {x: n for n, x in enumerate(np.unique(x_actual))}

# make co-occurrence plot - actual
cf_mat = np.ones((prep.num_types, len(x2x))) * 1e-9
for xi, yi in zip(x_actual, y_actual):
    cf_mat[yi, x2x[xi]] += 1
fig, ax = make_example_fig(np.log(cf_mat[:NUM_ROWS]))
plt.title('Actual')
plt.show()

# make joint plot - better but slow
x = []
y = []
x_actual_set = np.unique(x_actual)
y_actual_set = np.unique(y_actual)
cf_mat = np.ones((prep.num_types, len(x2x))) * 1e-9
for xia in x_actual:
    xi = np.random.choice(x_actual_set, size=1).item()
    yi = np.random.choice(y_actual_set, size=1).item()
    cf_mat[yi, x2x[xi]] += 1
fig, ax = make_example_fig(np.log(cf_mat[:NUM_ROWS]))

plt.title('Better but slow')
plt.show()


# make joint plot - better and fast
x = []
y = []
x_actual_set = np.unique(x_actual)
y_actual_set = np.unique(y_actual[:100])  # make next-word population small
cf_mat = np.ones((prep.num_types, len(x2x))) * 1e-9
for xia in x_actual:
    xi = np.random.choice(x_actual_set, size=1).item()
    yi = np.random.choice(y_actual_set, size=1).item()
    cf_mat[yi, x2x[xi]] += 1
fig, ax = make_example_fig(np.log(cf_mat[:NUM_ROWS]))

plt.title('Better and fast')
plt.show()


# make joint plot - worst
x = []
y = []
x_actual_set = np.unique(x_actual)
y_actual_set = np.unique(y_actual)
cf_mat = np.ones((prep.num_types, len(x2x))) * 1e-9
for xi in range(len(x2x)):
    yi = xi
    cf_mat[yi, xi] += 1
fig, ax = make_example_fig(np.log(cf_mat[:NUM_ROWS]))

plt.title('Worst case')
plt.show()