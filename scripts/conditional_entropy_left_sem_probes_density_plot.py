

from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# make binary
windows[:, -2] = [1 if prep.store.types[i] in probes else 0 for i in windows[:, -2]]
print(probes)
print(windows[:, -2])

x1, x2 = np.array_split(windows[:, -3], 2, axis=0)  # prev-word
y1, y2 = np.array_split(windows[:, -2], 2, axis=0)  # CAT member

g1 = sns.jointplot(x1, y1, kind='hex', ratio=1)
g2 = sns.jointplot(x2, y2, kind='hex', ratio=1)

plt.show()


