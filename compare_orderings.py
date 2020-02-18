from scipy.stats import spearmanr

from preppy import PartitionedPrep
from preppy.docs import load_docs

from ordermatters import config
from ordermatters.reorder import reorder_by_conditional_entropy
from ordermatters.reorder import reorder_by_joint_entropy

NUM_PARTS = 32
CORPUS_NAME = 'childes-20191206'
# PROBES_NAME = 'verbs-1321'
# PROBES_NAME = 'sem-4096'
PROBES_NAME = 'nouns-2972'
NUM_SKIP_FIRST_DOCS = 0

corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
docs, _ = load_docs(corpus_path, num_test_docs=0)
prep = PartitionedPrep(docs[NUM_SKIP_FIRST_DOCS:],
                       reverse=False,
                       num_types=4096,
                       num_parts=NUM_PARTS,
                       num_iterations=[1, 1],
                       batch_size=64,
                       context_size=2)

# load a category of words, X, for which to compute conditional entropy, H(X|Y)
probes_file_path = config.Dirs.words / f'{CORPUS_NAME}-{PROBES_NAME}.txt'
probes = [w for w in probes_file_path.read_text().split('\n') if w in prep.store.w2id]
print('num probes', len(probes))

# probes.remove('one')  # TODO test

# input to spearman correlation
ordered_part_ids = [n for n in range(prep.num_parts)]
reordered_part_ids_ce = reorder_by_conditional_entropy(prep, probes)
reordered_part_ids_je = reorder_by_joint_entropy(prep, probes)

print('ordering by decreasing conditional entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ce)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print('ordering by decreasing joint entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_je)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')