from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs
from categoryeval.probestore import ProbeStore

from ordermatters import configs
from ordermatters.figs import make_example_fig

CORPUS_NAME = 'newsela'
# CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'sem-4096'
NUM_TICKS = 2
NUM_TYPES = 4096 * 4 if CORPUS_NAME == 'newsela' else 4096
REMOVE_SYMBOLS = None

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
print(f'num probes={len(probes)}')

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False).copy()
print(f'Matrix containing all windows has shape={windows.shape}')

# probe windows
row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
probe_windows = windows[row_ids]
print(f'num probe windows={len(probe_windows)}')

probe_windows1, probe_windows2 = np.array_split(probe_windows[:, -2:], 2, axis=0)
print(probe_windows1.shape)  # (num windows, 2)  
print(probe_windows2.shape)  # (num windows, 2)


# init co-occurrence matrices
unique_x_ids1 = np.unique(probe_windows1[:, 0])
unique_y_ids1 = np.unique(probe_windows1[:, 1])
unique_x_ids2 = np.unique(probe_windows2[:, 0])
unique_y_ids2 = np.unique(probe_windows2[:, 1])
mat1 = np.zeros((len(unique_x_ids1), len(unique_y_ids1)))
mat2 = np.zeros((len(unique_x_ids2), len(unique_y_ids2)))
print(mat1.shape)  # (num x, num y)
print(mat2.shape)  # (num x, num y)

# make co-occurrence matrices
for pws, u_x_ids, u_y_ids, m in zip(
        [probe_windows1, probe_windows2],
        [unique_x_ids1, unique_x_ids2],
        [unique_y_ids1, unique_y_ids2],
        [mat1, mat2]):
    # original vocab id -> id with smaller range
    xi2xi = {original_id: n for n, original_id in enumerate(u_x_ids)}
    yi2yi = {original_id: n for n, original_id in enumerate(u_y_ids)}

    # collect co-occurrences
    for row in pws:
        m[xi2xi[row[0]], yi2yi[row[1]]] += 1

    print(f'sum={np.sum(m)}')

# plot co-occurrence matrices
fig1, ax1 = make_example_fig(np.log(mat1.T))
fig2, ax2 = make_example_fig(np.log(mat2.T))
ax1.set_title(f'{CORPUS_NAME} Partition 1')
ax2.set_title(f'{CORPUS_NAME} Partition 2')
plt.show()
