from scipy.stats import spearmanr

from preppy import PartitionedPrep

from ordermatters.toy_corpus import ToyCorpus
from ordermatters.reorder import reorder_by_conditional_entropy
from ordermatters.reorder import reorder_by_joint_entropy

NUM_PARTS = 32
NUM_TYPES = 4096

tc = ToyCorpus(num_parts=NUM_PARTS, num_types=NUM_TYPES)
probes = tc.nouns
prep = PartitionedPrep(tc.docs,
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=NUM_PARTS,
                       num_iterations=[1, 1],
                       batch_size=64,
                       context_size=1)

# input to spearman correlation
ordered_part_ids = [n for n in range(NUM_PARTS)]
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