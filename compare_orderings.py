from scipy.stats import spearmanr

from preppy import PartitionedPrep
from preppy.docs import load_docs

from ordermatters import config
from ordermatters.reorder import reorder_by_conditional_entropy

NUM_PARTS = 32
CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'nouns-2972'


corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
docs, _ = load_docs(corpus_path, num_test_docs=0)
prep = PartitionedPrep(docs,
                       reverse=False,
                       num_types=None,
                       num_parts=NUM_PARTS,
                       num_iterations=[1, 1],
                       batch_size=64,
                       context_size=2)

# load a category of words, X, for which to compute conditional entropy, H(X|Y)
probes_file_path = config.Dirs.words / f'{CORPUS_NAME}-{PROBES_NAME}.txt'
probes = [w for w in probes_file_path.read_text().split('\n') if w in prep.store.w2id]

# input to spearman correlation
ordered_part_ids = [n for n in range(prep.num_parts)]
reordered_part_ids = reorder_by_conditional_entropy(prep, probes)
print(reordered_part_ids)


# noinspection PyTypeChecker
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids)
print(f'rho={rho:.4f}')
print(f'p-v={p_value:.4f}')