import joblib
from collections import Counter
from numpy.lib.stride_tricks import as_strided
import numpy as np
from mpl_toolkits import mplot3d  # this is unused but needed for 3d plotting
import matplotlib.pyplot as plt

from preppy import PartitionedPrep
from preppy.docs import load_docs
from categoryeval.probestore import ProbeStore

from ordermatters import config

CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'sem-4096'
NUM_TICKS = 2
NUM_TYPES = 4096
REMOVE_SYMBOLS = ['.', '?']

corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path,
                          remove_symbols=REMOVE_SYMBOLS)

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
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False).copy()
print(f'Matrix containing all windows has shape={windows.shape}')

# probe windows
row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
probe_windows = windows[row_ids]
print(f'num probe windows={len(probe_windows)}')

xy1, xy2 = np.array_split(probe_windows[:, -3:-1], 2, axis=0)  # prev-word + cat-member
print(xy1.shape)
print(xy2.shape)

# z
row_hashes1 = [joblib.hash(row) for row in xy1]
row_hashes2 = [joblib.hash(row) for row in xy2]
row_hash2freq1 = Counter(row_hashes1)
row_hash2freq2 = Counter(row_hashes2)
z1 = np.log([row_hash2freq1[h] for h in row_hashes1])
z2 = np.log([row_hash2freq2[h] for h in row_hashes2])

fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('AO-CHILDES Partition 1')
ax.scatter3D(xy1[:, 0],
             xy1[:, 1],
             z1,
             c=z1,
             cmap='hsv')
ax.set_xlabel('previous word')
ax.set_ylabel('category word')
ax.set_zlim([0, 8])

fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('AO-CHILDES Partition 2')
ax.scatter3D(xy2[:, 0],
             xy2[:, 1],
             z2,
             c=z2,
             cmap='hsv')
ax.set_xlabel('previous word')
ax.set_ylabel('category word')
ax.set_zlim([0, 8])


plt.show()


