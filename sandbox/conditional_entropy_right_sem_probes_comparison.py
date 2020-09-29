

from numpy.lib.stride_tricks import as_strided
import numpy as np
from pyitlib import discrete_random_variable as drv

from preppy import PartitionedPrep
from preppy.docs import load_docs


from ordermatters import configs

CORPUS_NAME = 'newsela'
# CORPUS_NAME = 'childes-20191206'
WORDS_NAME = 'sem-4096'
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

test_words = (configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt').open().read().split("\n")
test_word_ids = [prep.store.w2id[w] for w in test_words if w in prep.store.w2id]  # must not be  a set
print(f'Including {len(test_word_ids)} out of {len(test_words)} test_words in file')

# windows
token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
shape = (num_possible_windows, prep.num_tokens_in_window)
windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
print(f'Matrix containing all windows has shape={windows.shape}')

# TODO the question: are the two methods equivalent: 1. nouns are binary, 2. nouns are not binary

###############
# METHOD 1: use all windows, and make noun observations binary
###############

x1, x2 = np.array_split(windows[:, -2], 2, axis=0)  # CAT member
y1, y2 = np.array_split(windows[:, -1], 2, axis=0)  # next-word

# make binary
x1 = [1 if prep.store.types[i] in test_words else 0 for i in x1]
x2 = [1 if prep.store.types[i] in test_words else 0 for i in x2]

ce1 = drv.entropy_conditional(x1, y1)
ce2 = drv.entropy_conditional(x2, y2)
print('method 1')
print(ce1)
print(ce2)

###############
# METHOD 2: use only probe windows, and leave noun observations non-binary
###############

# probe windows
row_ids = np.isin(windows[:, -2], test_word_ids)
probe_windows = windows[row_ids]
print(f'num probe windows={len(probe_windows)}')

x1, x2 = np.array_split(probe_windows[:, -2], 2, axis=0)  # CAT member
y1, y2 = np.array_split(probe_windows[:, -1], 2, axis=0)  # next-word

ce1 = drv.entropy_conditional(x1, y1)
ce2 = drv.entropy_conditional(x2, y2)
print('method 2')
print(ce1)
print(ce2)
