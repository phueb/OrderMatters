from typing import List

import numpy as np
from numpy.lib.stride_tricks import as_strided
from pyitlib import discrete_random_variable as drv

from preppy import PartitionedPrep


def reorder_by_conditional_entropy(prep: PartitionedPrep,
                                   probes: List[str],
                                   verbose: bool = False,
                                   ) -> List[int]:
    """
    1. compute conditional entropy of probe words, given distribution of words that follow them, for each partition.
    2. re-order partitions from high to low conditional entropy.
    """
    # calc H(X|Y) for each part
    ces = []
    for part in prep.reordered_parts:
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
        ces.append(drv.entropy_conditional(x, y).item())

    if verbose:
        print([round(ce, 2) for ce in ces])

    # get indices that sort conditional entropies from highest to lowest H(X|Y)
    res = list(sorted(range(prep.num_parts), key=lambda i: ces[i], reverse=True))
    return res


def reorder_by_joint_entropy(prep: PartitionedPrep,
                             probes: List[str],
                             verbose: bool = False,
                             ) -> List[int]:
    """
    1. compute conditional entropy of probe words, given distribution of words that follow them, for each partition.
    2. re-order partitions from high to low conditional entropy.
    """
    # calc H(X|Y) for each part
    jes = []
    for part in prep.reordered_parts:
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
        x_y = np.vstack((x, y))
        jes.append(drv.entropy_joint(x_y).item())

    if verbose:
        print([round(je, 2) for je in jes])

    # get indices that sort conditional entropies from highest to lowest H(X|Y)
    res = list(sorted(range(prep.num_parts), key=lambda i: jes[i], reverse=True))
    return res