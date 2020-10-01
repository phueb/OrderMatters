from scipy.stats import spearmanr

from preppy import PartitionedPrep

from ordermatters.corpus_toy import ToyCorpus
from ordermatters.reorder import compute_mutual_information_difference
from ordermatters.reorder import compute_joint_entropy

NUM_PARTS = 2
NUM_NOUNS = 512
MIN_NOUNS = 512  # this needs to be large to result in positive rho when sorting using conditional entropy
NUM_TYPES = 1024  # this needs to be large to result in positive rho when sorting using conditional entropy
DIVISOR = 8  # this needs to be large to result in positive rho when sorting using conditional entropy

tc = ToyCorpus(num_docs=NUM_PARTS,
               num_nouns=NUM_NOUNS,
               num_types=NUM_TYPES,
               divisor=DIVISOR,
               min_nouns=MIN_NOUNS,
               doc_offset=0,
               )
prep = PartitionedPrep(tc.docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=NUM_PARTS,
                       num_iterations=(1, 1),
                       batch_size=64,
                       context_size=1)

# input to spearman correlation
ordered_part_ids = [n for n in range(NUM_PARTS)]
test_words = [p for p in tc.nouns if p in prep.store.w2id]
reordered_part_ids_ce = compute_mutual_information_difference(prep, test_words, verbose=True)
reordered_part_ids_je = compute_joint_entropy(prep, test_words, verbose=True)

print('ordering by decreasing conditional entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_ce)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')

print('ordering by decreasing joint entropy:')
rho, p_value = spearmanr(ordered_part_ids, reordered_part_ids_je)
print(f'rho={rho: .4f}')
print(f'p-v={p_value: .4f}')