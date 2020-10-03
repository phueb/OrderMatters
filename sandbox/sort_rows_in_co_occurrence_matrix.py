"""
make algo that given co-occurrence matrix selects rows with highest cond entr vs least cond entropy
-> export a corpus with first in part1 and second in part 2
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple
import math
from pyitlib import discrete_random_variable as drv
from itertools import combinations

from ordermatters import configs
from ordermatters.memory import set_memory_limit
from ordermatters.utils import make_prep_from_naturalistic, make_windows, make_test_words

# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191206'
WORDS_NAME = 'sem-4096'
DISTANCE = + 1
REMOVE_SYMBOLS = None
REMOVE_NUMBERS = True

R = 4  # the smaller the faster
MIN_NW_F = 200  # the larger the faster
NUM_ROWS_MORE_OR_LESS = 10_000  # may not return results if this is too small

set_memory_limit(prop=0.9)


def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def process(_q, _iolock, _nw_id2row_ids, _test_word_windows):
    best_cei = 0.0
    best_test_word_ids_subset = None
    skipped = 0
    _total = 0
    while True:
        from_q: np.ndarray = _q.get()
        if from_q is None:

            # TODO save results - don't just print - put on q?

            with _iolock:
                print(f'Best test word ids subset, with ce={best_cei:.2f}:')
                print(best_test_word_ids_subset)
                print(f'skipped={skipped}/{_total}')
                print()
            break

        _total += 1

        # get only those test word windows that correspond to subset of given next-word ids
        _row_ids = []
        for nw_word_id in from_q:
            _row_ids.extend(_nw_id2row_ids[nw_word_id])

        # only get windows that result approximately in half of the number of total test word windows
        half = len(_test_word_windows) // 2
        if half - NUM_ROWS_MORE_OR_LESS < len(_row_ids) < half + NUM_ROWS_MORE_OR_LESS:
            rows = _test_word_windows[_row_ids]
        else:
            skipped += 1
            continue

        x = rows[:, -2]  # test_word
        y = rows[:, -2 + DISTANCE]  # left or right context
        cei = drv.entropy_conditional(x, y)

        if cei > best_cei:
            best_test_word_ids_subset = from_q
            best_cei = cei

        print(f'num rows={len(rows):,} | cond entropy={cei:.2f} ')


# get data
if DISTANCE != 1:
    raise NotImplementedError('if distance != 1, this would require modifying context_size.')
prep = make_prep_from_naturalistic(CORPUS_NAME, REMOVE_SYMBOLS, context_size=1)
test_words = make_test_words(prep, CORPUS_NAME, WORDS_NAME, REMOVE_NUMBERS)
test_word_ids = [prep.store.w2id[w] for w in test_words]
windows = make_windows(prep)
row_ids = np.isin(windows[:, -2], test_word_ids)
test_word_windows = windows[row_ids]

# exclude infrequent nws
next_word_ids = [i for i, f in zip(*np.unique(test_word_windows[:, -2 + DISTANCE], return_counts=True))
                 if f > MIN_NW_F]
row_ids = np.isin(test_word_windows[:, -2 + DISTANCE], next_word_ids)
test_word_windows = test_word_windows[row_ids]

print(f'Number of test-word types={len(test_word_ids):,}')
print(f'Number of next-word types={len(next_word_ids):,}')
print(f'Number of next-word rows ={len(test_word_windows):,}')

# nw_word_id2row_ids - where do next-words occur in test_word_windows?
nw_id2row_ids = {nw_id: [] for nw_id in next_word_ids}
for row_id, nw_id in enumerate(test_word_windows[:, -2 + DISTANCE]):
    nw_id2row_ids[nw_id].append(row_id)

# set up parallel processes
q = mp.Queue(maxsize=configs.Constants.num_processes)
iolock = mp.Lock()
pool = mp.Pool(configs.Constants.num_processes,
               initializer=process,
               initargs=(q, iolock, nw_id2row_ids, test_word_windows))
# raise SystemExit

# collect results in parallel
total = nCr(len(next_word_ids), R)
for n, nw_word_id_subset in enumerate(combinations(next_word_ids, R)):
    q.put(nw_word_id_subset)  # blocks until q below its max size
    print(f'{n:>24,}')
    print(f'{total:>24,}')


# tell workers we're done
for _ in range(configs.Constants.num_processes):
    q.put(None)
pool.close()
pool.join()

# for reference
print()
print('Test words + IDs')
for nwi, row_ids in nw_id2row_ids.items():
    w = prep.store.types[nwi]
    print(f'{nwi:>4} {w:>16} f={len(row_ids):>6,}')

# TODO export corpus