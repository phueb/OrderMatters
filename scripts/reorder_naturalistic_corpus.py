from scipy.stats import spearmanr
import numpy as np
from numpy.lib.stride_tricks import as_strided

from preppy import PartitionedPrep
from preppy.docs import load_docs

from ordermatters import configs
from ordermatters.reorder import compute_mutual_information_difference
from ordermatters.reorder import compute_joint_entropy
from ordermatters.reorder import compute_y_entropy
from ordermatters.reorder import compute_unconditional_entropy
from ordermatters.reorder import compute_information_interaction

NUM_PARTS = 32
# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191206'
REMOVE_NUMBER_WORDS = True  # this is important
NUM_SKIP_FIRST_DOCS = 0
RIGHT_NEIGHBORS = True

# WORDS_NAME = 'verbs-1321'
# WORDS_NAME = 'nouns-2972'
WORDS_NAME = 'sem-4096'
# WORDS_NAME = 'adjs-498'

corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
docs, _ = load_docs(corpus_path, num_test_docs=0)
prep = PartitionedPrep(docs[NUM_SKIP_FIRST_DOCS:],
                       reverse=False,
                       num_types=4096 * 4 if CORPUS_NAME == 'newsela' else 4096,
                       num_parts=NUM_PARTS,
                       num_iterations=(1, 1),
                       batch_size=64,
                       context_size=2)  # need a window of size=3 for inf-interaction

# load a category of words, X, for which to compute conditional entropy, H(X|Y)
test_words_file_path = configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt'
test_words = [w for w in test_words_file_path.read_text().split('\n') if w in prep.store.w2id]
test_word_ids = [prep.store.w2id[w] for w in test_words]
print('num test_words', len(test_words))

if REMOVE_NUMBER_WORDS:  # number words are not nouns
    number_words_file_path = configs.Dirs.words / f'{CORPUS_NAME}-numbers.txt'
    for number in [w for w in number_words_file_path.read_text().split('\n')]:
        if number in test_words:
            test_words.remove(number)
            print('Removed', number)

if RIGHT_NEIGHBORS:
    col_id = -1
    direction = 'right'
else:
    col_id = -3
    direction = 'left'

print(f'Using {direction} neighbors')

# init
jes = []
yes = []
iis = []
mds = []
ues = []

# for each part, compute multiple measures
for n, part in enumerate(prep.reordered_parts):
    print(f'part={n+1}/{NUM_PARTS}')

    # make windows once, to be used by each measure
    token_ids_array = part.astype(np.int64)
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

    # windows with test_word in position -2
    row_ids = np.isin(windows[:, -2], test_word_ids)
    test_word_windows = windows[row_ids]

    # compute measures
    jes.append(compute_joint_entropy(test_word_windows, col_id))
    yes.append(compute_y_entropy(test_word_windows, col_id))
    ues.append(compute_unconditional_entropy(windows, col_id))
    iis.append(compute_information_interaction(test_word_windows))  # bidirectional
    mds.append(compute_mutual_information_difference(windows, test_word_windows, col_id))

ordered_part_ids = [n for n in range(prep.num_parts)]

# get indices that sort above measures
reordered_part_ids_je = np.fromiter(sorted(range(prep.num_parts), key=lambda i: jes[i], reverse=False), dtype=np.float)
reordered_part_ids_ye = np.fromiter(sorted(range(prep.num_parts), key=lambda i: yes[i], reverse=False), dtype=np.float)
reordered_part_ids_ue = np.fromiter(sorted(range(prep.num_parts), key=lambda i: ues[i], reverse=False), dtype=np.float)
reordered_part_ids_ii = np.fromiter(sorted(range(prep.num_parts), key=lambda i: iis[i], reverse=False), dtype=np.float)
reordered_part_ids_md = np.fromiter(sorted(range(prep.num_parts), key=lambda i: mds[i], reverse=False), dtype=np.float)

print(f'ordering by increasing joint entropy of {direction} test-words + neighbors:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_je)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print(f'ordering by increasing entropy of {direction} test-word neighbors:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ye)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print(f'ordering by increasing all-word entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ue)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print(f'ordering by increasing information-interaction between test-word and right+left neighbors:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ii)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print(f'ordering by increasing mutual info difference between test-word and {direction} neighbor:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_md)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')