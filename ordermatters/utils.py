from numpy.lib.stride_tricks import as_strided
from typing import List, Tuple, Optional
import numpy as np

from preppy import PartitionedPrep
from preppy.docs import load_docs

from ordermatters import configs


def make_prep(corpus_name: str,
              remove_symbols: Optional[List[str]] = None,
              context_size: int = 7,
              ) -> PartitionedPrep:
    corpus_path = configs.Dirs.corpora / f'{corpus_name}.txt'
    train_docs, _ = load_docs(corpus_path,
                              remove_symbols=remove_symbols)

    prep = PartitionedPrep(train_docs,
                           reverse=False,
                           num_types=4096 * 4 if corpus_name == 'newsela' else 4096,
                           num_parts=2,
                           num_iterations=(20, 20),
                           batch_size=64,
                           context_size=context_size,
                           )
    return prep


def make_test_words(prep: PartitionedPrep,
                    corpus_name: str,
                    words_name: str,
                    remove_numbers: bool = True) -> List[str]:
    number_words = (configs.Dirs.words / f'{corpus_name}-numbers.txt').open().read().split("\n")
    test_words_all = (configs.Dirs.words / f'{corpus_name}-{words_name}.txt').open().read().split("\n")

    test_words = []
    for w in test_words_all:
        if remove_numbers and w in number_words:
            continue
        if w in prep.store.w2id:
            test_words.append(w)

    print(f'Including {len(test_words)} out of {len(test_words_all)} test_words in file')

    return test_words


def make_windows(prep):
    token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
    print(f'Matrix containing all windows has shape={windows.shape}')

    return windows