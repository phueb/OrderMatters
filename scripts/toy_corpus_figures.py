import pyitlib.discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep

from ordermatters.figs import make_example_fig
from ordermatters.toy_corpus import ToyCorpus

NUM_PARTS = 2
NUM_NOUNS = 512
MIN_NOUNS = 512  # this needs to be large to result in positive rho when sorting using conditional entropy
NUM_TYPES = 1024  # this needs to be large to result in positive rho when sorting using conditional entropy
DIVISOR = 8

tc = ToyCorpus(num_docs=NUM_PARTS,
               num_nouns=NUM_NOUNS,
               num_types=NUM_TYPES,
               divisor=DIVISOR,
               min_nouns=MIN_NOUNS,
               doc_offset=2,
               )
prep = PartitionedPrep(tc.docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=NUM_PARTS,
                       num_iterations=[1, 1],
                       batch_size=64,
                       context_size=1)
probes = [p for p in tc.nouns if p in prep.store.w2id]

for part_id, part in enumerate(prep.reordered_parts):

    # windows
    token_ids_array = part.astype(np.int64)
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

    # windows with probe in position -2
    row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
    probe_windows = windows[row_ids]

    # conditional entropy
    x = probe_windows[:, -2]  # CAT member
    y = probe_windows[:, -1]  # next-word

    # map word ID of nouns to IDs between [0, len(probes)]
    # this makes creating a matrix with the right number of columns easier
    x2x = {xi: n for n, xi in enumerate(np.unique(x))}

    # make co-occurrence plot
    cf_mat1 = np.ones((prep.num_types, len(x2x))) * 1e-9
    for xi, yi in zip(x, y):
        cf_mat1[yi, x2x[xi]] += 1
    last_num_rows = NUM_TYPES - NUM_NOUNS  # other rows are just empty because of nouns not occurring with nouns
    fig, ax = make_example_fig(np.log(cf_mat1[-last_num_rows:]))
    ce = drv.entropy_conditional(x, y).item()
    plt.title(f'Toy Corpus Part {part_id}\nH(X|Y)={ce:.4f}')
    plt.show()

    print(np.sum(cf_mat1))

