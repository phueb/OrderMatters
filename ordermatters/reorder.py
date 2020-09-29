from typing import List

import numpy as np
from numpy.lib.stride_tricks import as_strided
from pyitlib import discrete_random_variable as drv

from preppy import PartitionedPrep


def reorder_by_conditional_entropy(prep: PartitionedPrep,
                                   test_words: List[str],
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
        row_ids = np.isin(windows[:, -2], test_word_ids)
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
                             test_words: List[str],
                             verbose: bool = False,
                             ) -> List[int]:
    """
    1. compute conditional entropy of probe words, given distribution of words that follow them, for each partition.
    2. re-order partitions from low to high joint entropy.
    """
    # calc H(X,Y) for each part
    jes = []
    for part in prep.reordered_parts:
        # windows
        token_ids_array = part.astype(np.int64)
        num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
        shape = (num_possible_windows, prep.num_tokens_in_window)
        windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

        # windows with probe in position -2
        row_ids = np.isin(windows[:, -2], test_word_ids)
        probe_windows = windows[row_ids]

        # joint entropy
        x = probe_windows[:, -2]  # CAT member
        y = probe_windows[:, -1]  # next-word
        x_y = np.vstack((x, y))
        jes.append(drv.entropy_joint(x_y).item())

    if verbose:
        print([round(je, 2) for je in jes])

    # get indices that sort joint entropies from lowest to highest
    res = list(sorted(range(prep.num_parts), key=lambda i: jes[i], reverse=False))
    return res


def reorder_by_y_entropy(prep: PartitionedPrep,
                         test_words: List[str],
                         verbose: bool = False,
                         ) -> List[int]:
    """
    1. compute conditional entropy of probe words, given distribution of words that follow them, for each partition.
    2. re-order partitions from low to high Y entropy.
    """
    # calc H(Y) for each part
    yes = []
    for part in prep.reordered_parts:
        # windows
        token_ids_array = part.astype(np.int64)
        num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
        shape = (num_possible_windows, prep.num_tokens_in_window)
        windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

        # windows with probe in position -2
        row_ids = np.isin(windows[:, -2], test_word_ids)
        probe_windows = windows[row_ids]

        # entropy of Y
        y = probe_windows[:, -1]  # next-word
        yes.append(drv.entropy_joint(y).item())

    if verbose:
        print([round(je, 2) for je in yes])

    # get indices that sort Y entropies from low to high
    res = list(sorted(range(prep.num_parts), key=lambda i: yes[i], reverse=False))
    return res


def reorder_by_unconditional_entropy(prep: PartitionedPrep,
                                     test_words: List[str],
                                     verbose: bool = False,
                                     ) -> List[int]:
    """
    1. compute entropy of all words in a part
    2. re-order partitions from low to high entropy.
    """
    # calc H(Y) for each part
    yes = []
    for part in prep.reordered_parts:
        # windows
        token_ids_array = part.astype(np.int64)
        num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
        shape = (num_possible_windows, prep.num_tokens_in_window)
        windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

        # entropy of all windows
        y = windows[:, -1]
        yes.append(drv.entropy_joint(y).item())

    if verbose:
        print([round(je, 2) for je in yes])

    # get indices that sort Y entropies from low to high
    res = list(sorted(range(prep.num_parts), key=lambda i: yes[i], reverse=False))
    return res


def reorder_by_information_interaction(prep: PartitionedPrep,
                                       test_words: List[str],
                                       verbose: bool = False,
                                       ) -> List[int]:
    """
    1. compute info-interaction between probe, left-word, and right-word probability distributions
    2. re-order partitions from low to high.
    """
    # calc H(Y) for each part
    iis = []
    for part in prep.reordered_parts:
        # windows
        token_ids_array = part.astype(np.int64)
        num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
        shape = (num_possible_windows, prep.num_tokens_in_window)
        windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

        # windows with probe in position -2
        row_ids = np.isin(windows[:, -2], test_word_ids)
        probe_windows = windows[row_ids]

        # info interaction
        x = probe_windows[:, -3:].T  # 3 rows, where each row contains realisations of a distinct random variable
        iis.append(drv.information_interaction(x))

    if verbose:
        print([round(je, 2) for je in iis])

    # get indices that sort ii from low to high
    res = list(sorted(range(prep.num_parts), key=lambda i: iis[i], reverse=False))
    return res