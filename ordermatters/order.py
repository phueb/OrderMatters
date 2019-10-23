import numpy as np


def order_pseudo_randomly(self, parts):
    idx = []
    for i, j in zip(np.roll(np.arange(self.params.num_parts), self.params.num_parts // 2)[::2],
                    np.roll(np.arange(self.params.num_parts), self.params.num_parts // 2)[::-2]):
        idx += [i, j]
    assert len(set(idx)) == len(parts)
    res = [parts[i] for i in idx]
    assert len(res) == len(parts)
    return res