from scipy.stats import spearmanr

from preppy import PartitionedPrep
from preppy.docs import load_docs

from ordermatters import configs
from ordermatters.reorder import reorder_by_conditional_entropy
from ordermatters.reorder import reorder_by_joint_entropy
from ordermatters.reorder import reorder_by_y_entropy
from ordermatters.reorder import reorder_by_unconditional_entropy
from ordermatters.reorder import reorder_by_information_interaction

NUM_PARTS = 32
CORPUS_NAME = 'newsela'
# CORPUS_NAME = 'childes-20191206'
REMOVE_NUMBER_WORDS = True  # this is important
NUM_SKIP_FIRST_DOCS = 0

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
                       context_size=1)

# load a category of words, X, for which to compute conditional entropy, H(X|Y)
test_words_file_path = configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt'
test_words = [w for w in test_words_file_path.read_text().split('\n') if w in prep.store.w2id]
print('num test_words', len(test_words))

if REMOVE_NUMBER_WORDS:  # number words are not nouns
    number_words_file_path = configs.Dirs.words / f'{CORPUS_NAME}-numbers.txt'
    for number in [w for w in number_words_file_path.read_text().split('\n')]:
        if number in test_words:
            test_words.remove(number)
            print('Removed', number)


# input to spearman correlation
ordered_part_ids = [n for n in range(prep.num_parts)]
reordered_part_ids_ce = reorder_by_conditional_entropy(prep, test_words)
reordered_part_ids_je = reorder_by_joint_entropy(prep, test_words)
reordered_part_ids_ye = reorder_by_y_entropy(prep, test_words)
reordered_part_ids_ue = reorder_by_unconditional_entropy(prep, test_words)
reordered_part_ids_ii = reorder_by_information_interaction(prep, test_words)

print('ordering by decreasing conditional entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ce)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print('ordering by increasing joint entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_je)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print('ordering by increasing Y entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ye)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print('ordering by increasing unconditional entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ue)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print('ordering by increasing information-interaction:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ii)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')